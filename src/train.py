"""Train FastWAM on LIBERO.

Quick sanity check (should drive the loss toward 0 on one repeated batch):
    CUDA_VISIBLE_DEVICES=1 python train.py --overfit --steps 300 --batch-size 4

Small generalisation test (few training episodes, judged on held-out episodes):
    CUDA_VISIBLE_DEVICES=1 python train.py --max-train-episodes 60 --steps 3000 --out runs/subset60

Real run:
    CUDA_VISIBLE_DEVICES=1 python train.py --suites libero_spatial --steps 20000 --batch-size 8
"""
import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch

from src.data import load_data, make_loader
from src.models import load_dit, load_text_encoder, load_vae
from src.wam import FastWAM

CAMS = ("observation.images.image", "observation.images.wrist_image")
TEXT_LEN = 512                                   # Wan's T5 context length
LOG_FIELDS = ["step", "loss", "loss_act", "loss_video", "lr", "grad_norm", "sec_per_step",
              "train_loss_act", "train_action_l1",
              "val_loss_act", "val_loss_video", "val_action_l1", "baseline_action_l1"]


# ---------------------------------------------------------------- encoders (frozen, outside the model)
@torch.no_grad()
def encode_text(tok, t5, prompts, device):
    """Same recipe as diffusers' WanPipeline: pad to 512, zero out positions past the prompt."""
    inputs = tok(prompts, padding="max_length", max_length=TEXT_LEN, truncation=True,
                 add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
    ids, mask = inputs.input_ids.to(device), inputs.attention_mask.to(device)
    emb = t5(ids, mask).last_hidden_state
    return emb * mask.unsqueeze(-1).to(emb.dtype)                        # (N, 512, 4096)


def build_text_cache(prompts, device):
    """Embed every task instruction once, then free the T5 encoder (~11 GB)."""
    tok, t5 = load_text_encoder(device)
    prompts = sorted(set(prompts))
    embs = encode_text(tok, t5, prompts, device)
    del tok, t5
    torch.cuda.empty_cache()
    return {p: e for p, e in zip(prompts, embs)}


@torch.no_grad()
def encode_video(vae, batch, device, chunk=2):
    """Both cameras side by side -> Wan VAE -> normalised latents (B, 16, 1+T//4, H/8, 2W/8)."""
    video = torch.cat([batch[k] for k in CAMS], dim=-1)                  # (B, T, 3, H, 2W) in [0, 1]
    video = video.permute(0, 2, 1, 3, 4) * 2 - 1                          # (B, 3, T, H, 2W) in [-1, 1]
    mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
    zs = []
    for part in video.split(chunk):                                       # small chunks bound VAE memory
        z = vae.encode(part.to(device, vae.dtype)).latent_dist.mode()
        zs.append((z.float() - mean) / std)
    return torch.cat(zs)


# ---------------------------------------------------------------- helpers
def combined_stats(stats, key):
    """Pool per-suite mean/std (weighted by count) into one mean/std."""
    parts = [s[key] for s in stats.values()]
    n = np.array([float(np.asarray(p["count"]).reshape(-1)[0]) for p in parts])
    means = np.stack([np.asarray(p["mean"], dtype=np.float64) for p in parts])
    vars_ = np.stack([np.asarray(p["std"], dtype=np.float64) ** 2 for p in parts])
    mean = (n[:, None] * means).sum(0) / n.sum()
    var = (n[:, None] * (vars_ + (means - mean) ** 2)).sum(0) / n.sum()
    return mean, np.sqrt(var)


def lr_at(step, base_lr, warmup, total):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def batch_inputs(batch, vae, text_cache, device):
    z = encode_video(vae, batch, device)
    text = torch.stack([text_cache[t] for t in batch["task"]]).float()
    actions = batch["action"].to(device).float()
    state = batch["observation.state"].to(device).float()
    act_pad = batch["action_is_pad"].to(device)
    return z, text, actions, state, act_pad


@torch.no_grad()
def evaluate(model, batches, vae, text_cache, device):
    """On fixed batches, with fixed noise so evals are comparable over time:
       loss_act / loss_video : the training losses
       action_l1             : |sampled actions - true actions| in robot units (real inference path)
       baseline_l1           : same error for always predicting the dataset-mean action
    """
    model.eval()
    rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    torch.manual_seed(0)
    loss_act, loss_vid, l1, base = [], [], [], []
    for batch in batches:
        z, text, actions, state, act_pad = batch_inputs(batch, vae, text_cache, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, parts = model.loss(z, text, actions, state, act_pad)
            pred = model.sample_actions(z[:, :, :model.n_cur_frames], text, state, steps=10)
        keep = ~act_pad
        loss_act.append(parts["loss_act"].item())
        loss_vid.append(parts["loss_video"].item())
        l1.append((pred.float() - actions).abs().mean(-1)[keep].mean().item())
        base.append((model.action_mean - actions).abs().mean(-1)[keep].mean().item())
    torch.set_rng_state(rng[0]); torch.cuda.set_rng_state(rng[1])
    model.train()
    return {"loss_act": np.mean(loss_act), "loss_video": np.mean(loss_vid),
            "action_l1": np.mean(l1), "baseline_l1": np.mean(base)}


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suites", nargs="+", default=["libero_spatial"])
    p.add_argument("--max-train-episodes", type=int, default=None,
                   help="train on only this many episodes (held-out val episodes are unchanged)")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--lam-video", type=float, default=1.0)
    p.add_argument("--act-hidden", type=int, default=512)
    p.add_argument("--chunk", type=int, default=16)
    p.add_argument("--n-future", type=int, default=16, help="future frames; must be a multiple of 4")
    p.add_argument("--freeze-dit", action="store_true", help="train only the action expert")
    p.add_argument("--no-grad-ckpt", action="store_true", help="faster, but much more memory")
    p.add_argument("--overfit", action="store_true", help="train on one repeated batch (sanity check)")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=4, help="fixed batches per eval set")
    p.add_argument("--save-every", type=int, default=2000, help="0 = never save")
    p.add_argument("--out", default="runs/fastwam")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    assert args.n_future % 4 == 0, "Wan's VAE packs 4 frames per latent frame"

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # ---- data
    common = dict(chunk=args.chunk, n_future=args.n_future)
    train_ds, stats = load_data(args.suites, split="train", max_episodes=args.max_train_episodes, **common)
    val_ds, _ = load_data(args.suites, split="val", **common)
    train_dl = make_loader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers)
    val_dl = make_loader(val_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    n_eps = sum(d.meta.total_episodes if d.episodes is None else len(d.episodes)
                for d in (train_ds.datasets if hasattr(train_ds, "datasets") else [train_ds]))
    print(f"train: {len(train_ds)} samples from {n_eps} episodes | val: {len(val_ds)} samples (held out)")

    # ---- frozen encoders
    tasks = [t for d in (train_ds.datasets if hasattr(train_ds, "datasets") else [train_ds])
             for t in d.meta.tasks.index]
    text_cache = build_text_cache(tasks, device)
    vae = load_vae(device)
    print(f"cached {len(text_cache)} task embeddings")

    # ---- model: fp32 master weights, bf16 autocast for compute
    model = FastWAM(load_dit(device), chunk=args.chunk, act_hidden=args.act_hidden,
                    lam_video=args.lam_video).to(device).float()
    model.grad_ckpt = not args.no_grad_ckpt
    a_mean, a_std = combined_stats(stats, "action")              # full-suite stats (fixed, not a subset)
    s_mean, s_std = combined_stats(stats, "observation.state")
    model.set_norm_stats(a_mean, a_std, s_mean, s_std)
    if args.freeze_dit:
        model.backbone.dit.requires_grad_(False)

    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params) / 1e6
    n_ae = sum(p.numel() for p in model.action_expert.parameters()) / 1e6
    print(f"trainable params: {n_train:.0f}M (action expert {n_ae:.0f}M)")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    # ---- batches: fixed eval sets drawn once (seeded), so eval numbers are comparable over time
    g = torch.Generator().manual_seed(1234)
    train_eval = [b for _, b in zip(range(args.eval_batches),
                                    make_loader(train_ds, args.batch_size, num_workers=args.num_workers,
                                                generator=g))]
    val_eval = [b for _, b in zip(range(args.eval_batches), val_dl)]
    if args.overfit:
        fixed = next(iter(train_dl))
        train_iter = iter(lambda: fixed, None)
        train_eval, val_eval = [fixed], [fixed]
    else:
        def forever():
            while True:
                yield from train_dl
        train_iter = forever()

    log_f = open(out / "log.csv", "w", newline="")
    log = csv.DictWriter(log_f, fieldnames=LOG_FIELDS)
    log.writeheader()

    model.train()
    t0 = time.time()
    for step in range(args.steps):
        lr = lr_at(step, args.lr, args.warmup, args.steps)
        for grp in opt.param_groups:
            grp["lr"] = lr

        batch = next(train_iter)
        z, text, actions, state, act_pad = batch_inputs(batch, vae, text_cache, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, parts = model.loss(z, text, actions, state, act_pad)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip).item()
        opt.step()

        row = {}
        if step % args.log_every == 0 or step == args.steps - 1:
            torch.cuda.synchronize()
            dt = (time.time() - t0) / (args.log_every if step else 1)
            t0 = time.time()
            mem = torch.cuda.max_memory_allocated() / 2**30
            print(f"step {step:6d} | loss {loss.item():.4f} (act {parts['loss_act'].item():.4f} "
                  f"video {parts['loss_video'].item():.4f}) | lr {lr:.2e} | gnorm {grad_norm:.2f} "
                  f"| {dt:.2f}s/step | mem {mem:.1f}GB", flush=True)
            row.update(step=step, loss=loss.item(), loss_act=parts["loss_act"].item(),
                       loss_video=parts["loss_video"].item(), lr=lr, grad_norm=grad_norm, sec_per_step=dt)

        if step % args.eval_every == 0 or step == args.steps - 1:
            tr = evaluate(model, train_eval, vae, text_cache, device)
            va = evaluate(model, val_eval, vae, text_cache, device)
            print(f"  eval @ {step}: action L1  train {tr['action_l1']:.4f} | held-out {va['action_l1']:.4f}"
                  f" | mean-action baseline {va['baseline_l1']:.4f}   "
                  f"(loss_act train {tr['loss_act']:.4f} held-out {va['loss_act']:.4f})", flush=True)
            row.update(step=step, train_loss_act=tr["loss_act"], train_action_l1=tr["action_l1"],
                       val_loss_act=va["loss_act"], val_loss_video=va["loss_video"],
                       val_action_l1=va["action_l1"], baseline_action_l1=va["baseline_l1"])
            t0 = time.time()                                       # don't count eval time as step time

        if row:
            log.writerow(row); log_f.flush()

        if args.save_every > 0 and ((step > 0 and step % args.save_every == 0) or step == args.steps - 1):
            torch.save({"model": model.state_dict(), "args": vars(args), "step": step},
                       out / f"ckpt_{step:06d}.pt")
            print(f"  saved {out / f'ckpt_{step:06d}.pt'}", flush=True)

    log_f.close()


if __name__ == "__main__":
    main()
