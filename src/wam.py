import torch.nn as nn 
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import math
from diffusers import AutoencoderKLWan, WanTransformer3DModel

class TimestepEmbedder(nn.Module):
    """Noise level t in [0, 1] -> vector of size `hidden` (sinusoidal features + small MLP)."""
    def __init__(self, hidden: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def forward(self, t):                                       # t: (B,)
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = (t.float() * 1000)[:, None] * freqs[None]        # [0,1] -> [0,1000] like DiT timesteps
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb.to(self.mlp[0].weight.dtype))       # (B, hidden)


class ActionBlock(nn.Module):
    """One action-expert layer, split around the shared attention.

    pre_attn : tokens (B, L, hidden) -> q, k, v (B, heads, L, head_dim) sized to match the video's
    post_attn: shared-attention output -> out proj, text cross-attention, MLP
    """
    def __init__(self, hidden: int, attn_dim: int, n_heads: int, text_dim: int,
                 mlp_ratio: int = 4, cross_heads: int = 8):
        super().__init__()
        assert hidden % cross_heads == 0, f"act_hidden={hidden} must be divisible by cross_heads={cross_heads}"
        self.n_heads = n_heads
        self.head_dim = attn_dim // n_heads

        # adaLN-Zero: t_emb -> shift/scale/gate for attention and MLP (6 values per channel)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 6 * hidden))
        nn.init.zeros_(self.adaLN[1].weight)
        nn.init.zeros_(self.adaLN[1].bias)

        # shared self-attention: project UP to the video's attention size, then back down
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.to_q = nn.Linear(hidden, attn_dim)
        self.to_k = nn.Linear(hidden, attn_dim)
        self.to_v = nn.Linear(hidden, attn_dim)
        self.norm_q = nn.RMSNorm(attn_dim, eps=1e-6)            # Wan normalises q/k across heads
        self.norm_k = nn.RMSNorm(attn_dim, eps=1e-6)
        self.to_out = nn.Linear(attn_dim, hidden)

        # cross-attention to the (projected) text
        self.norm2 = nn.LayerNorm(hidden, eps=1e-6)
        self.cross = nn.MultiheadAttention(hidden, num_heads=cross_heads, kdim=text_dim, vdim=text_dim, batch_first=True)

        # MLP
        self.norm3 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(hidden, mlp_ratio * hidden), nn.GELU(approximate="tanh"),
                                 nn.Linear(mlp_ratio * hidden, hidden))

    def _heads(self, x):                                        # (B, L, attn_dim) -> (B, heads, L, head_dim)
        return x.unflatten(2, (self.n_heads, self.head_dim)).transpose(1, 2)

    def pre_attn(self, x, t_emb):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN(t_emb).unsqueeze(1).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale1) + shift1
        q = self._heads(self.norm_q(self.to_q(h)))
        k = self._heads(self.norm_k(self.to_k(h)))
        v = self._heads(self.to_v(h))
        cache = (gate1, shift2, scale2, gate2)                  # needed after the attention
        return q, k, v, cache

    def post_attn(self, x, attn_out, cache, text):
        gate1, shift2, scale2, gate2 = cache
        attn_out = attn_out.transpose(1, 2).flatten(2)          # (B, L, attn_dim)
        x = x + gate1 * self.to_out(attn_out)
        x = x + self.cross(self.norm2(x), text, text, need_weights=False)[0]
        x = x + gate2 * self.mlp(self.norm3(x) * (1 + scale2) + shift2)
        return x


class ActionExpert(nn.Module):
    """The new, small transformer for the action chunk (the teal column).

    embed     : noisy actions + state + t_action -> tokens xa (B, chunk+1, hidden), t_emb (B, hidden)
    pre_attn  : layer i, tokens -> q, k, v, cache
    post_attn : layer i, shared-attention output -> updated tokens
    head      : tokens -> action velocity (B, chunk, action_dim), state slot dropped
    """
    def __init__(self, action_dim, state_dim, chunk, hidden, attn_dim, n_heads, n_layers, text_dim,
                 mlp_ratio=4, cross_heads=8):
        super().__init__()
        # embedding
        self.action_in = nn.Linear(action_dim, hidden)
        self.state_in  = nn.Linear(state_dim, hidden)
        self.act_pos   = nn.Parameter(torch.zeros(1, chunk + 1, hidden))   # +1 for the state token
        nn.init.normal_(self.act_pos, std=0.02)
        self.act_time  = TimestepEmbedder(hidden)

        # one block per DiT layer
        self.blocks = nn.ModuleList(
            [ActionBlock(hidden, attn_dim, n_heads, text_dim, mlp_ratio, cross_heads) for _ in range(n_layers)]
        )

        # output head (zero init: predicts 0 velocity at the start)
        self.norm_out = nn.LayerNorm(hidden)
        self.proj_out = nn.Linear(hidden, action_dim)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def embed(self, noisy_actions, state, t_action):
        a  = self.action_in(noisy_actions)                      # (B, chunk, hidden)
        s  = self.state_in(state).unsqueeze(1)                  # (B, 1, hidden)
        xa = torch.cat([s, a], dim=1) + self.act_pos            # (B, chunk+1, hidden)  [state, a0..]
        t_emb = self.act_time(t_action)                         # (B, hidden)
        return xa, t_emb

    def pre_attn(self, i, xa, t_emb):
        return self.blocks[i].pre_attn(xa, t_emb)

    def post_attn(self, i, xa, attn_out, cache, text):
        return self.blocks[i].post_attn(xa, attn_out, cache, text)

    def head(self, xa):
        return self.proj_out(self.norm_out(xa[:, 1:]))        

def apply_rope(x, freqs_cos, freqs_sin):
    """Wan's rotary embedding (copied from WanAttnProcessor). x: (B, L, heads, head_dim)."""
    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
    cos, sin = freqs_cos[..., 0::2], freqs_sin[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(x)


class DiTBackbone(nn.Module):
    """Adapter around the pretrained Wan DiT. Adds no new weights;
    it re-arranges WanTransformer3DModel.forward so actions can join each layer's attention.

    embed     : latents z + per-token timesteps + T5 text -> video tokens and conditioning
    pre_attn  : layer i, tokens -> q, k, v (B, heads, L, head_dim), cache
    post_attn : layer i, attention output -> out proj, text cross-attention, FFN
    head      : tokens -> velocity in latent-video shape
    """
    def __init__(self, dit: WanTransformer3DModel):
        super().__init__()
        self.dit = dit

    def tokens_per_frame(self, z_shape):
        _, _, _, H, W = z_shape
        _, p_h, p_w = self.dit.config.patch_size
        return (H // p_h) * (W // p_w)

    def embed(self, z, t_video, text, n_cur_frames=1):
        """z: (B, C, F, H, W), latent frame(s) 0 clean | t_video: (B,) in [0,1] | text: (B, 512, 4096)"""
        B, _, F, _, _ = z.shape
        p_t = self.dit.config.patch_size[0]
        n_cur = n_cur_frames * self.tokens_per_frame(z.shape)
        n_tok = (F // p_t) * self.tokens_per_frame(z.shape)

        # 1. positions (from z's shape)
        rope = self.dit.rope(z)

        # 2. latents -> tokens, frame-major: the first n_cur tokens are the current frame
        xv = self.dit.patch_embedding(z).flatten(2).transpose(1, 2)    # (B, n_tok, D)

        # 3. per-token timestep (Wan's 0-1000 scale): 0 for current frame, t_video for the future
        t_tok = (t_video.float() * 1000)[:, None].expand(B, n_tok).clone()
        t_tok[:, :n_cur] = 0

        # 4. time + text through Wan's condition embedder
        temb, timestep_proj, text_proj, _ = self.dit.condition_embedder(
            t_tok.flatten(), text, None, timestep_seq_len=n_tok
        )
        timestep_proj = timestep_proj.unflatten(2, (6, -1))           # (B, n_tok, 6, D)
        return xv, rope, timestep_proj, temb, text_proj

    def pre_attn(self, i, xv, timestep_proj, rope):
        blk, attn = self.dit.blocks[i], self.dit.blocks[i].attn1

        # this block's 6 modulation values per token: shared timestep_proj + block's own table
        mods = (blk.scale_shift_table.unsqueeze(0) + timestep_proj.float()).chunk(6, dim=2)
        shift, scale, gate, c_shift, c_scale, c_gate = (m.squeeze(2) for m in mods)

        h = (blk.norm1(xv.float()) * (1 + scale) + shift).type_as(xv)
        q, k, v = attn.to_q(h), attn.to_k(h), attn.to_v(h)
        q, k = attn.norm_q(q), attn.norm_k(k)
        q, k, v = (t.unflatten(2, (attn.heads, -1)) for t in (q, k, v))   # (B, L, heads, head_dim)
        q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))                 # (B, heads, L, head_dim)

        cache = (gate, c_shift, c_scale, c_gate)
        return q, k, v, cache

    def post_attn(self, i, xv, attn_out, cache, text_proj):
        blk, attn = self.dit.blocks[i], self.dit.blocks[i].attn1
        gate, c_shift, c_scale, c_gate = cache

        # finish self-attention: merge heads, out proj, gated residual
        out = attn_out.transpose(1, 2).flatten(2, 3).type_as(xv)         # (B, L, D)
        out = attn.to_out[1](attn.to_out[0](out))
        xv = (xv.float() + out * gate).type_as(xv)

        # cross-attention to text
        h = blk.norm2(xv.float()).type_as(xv)
        xv = xv + blk.attn2(h, text_proj, None, None)

        # feed-forward
        h = (blk.norm3(xv.float()) * (1 + c_scale) + c_shift).type_as(xv)
        xv = (xv.float() + blk.ffn(h).float() * c_gate).type_as(xv)
        return xv

    def head(self, xv, temb, z_shape):
        B, _, F, H, W = z_shape
        p_t, p_h, p_w = self.dit.config.patch_size

        shift, scale = (self.dit.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
        x = (self.dit.norm_out(xv.float()) * (1 + scale.squeeze(2)) + shift.squeeze(2)).type_as(xv)
        x = self.dit.proj_out(x)

        # unpatchify back to (B, C, F, H, W)
        x = x.reshape(B, F // p_t, H // p_h, W // p_w, p_t, p_h, p_w, -1)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return x.flatten(6, 7).flatten(4, 5).flatten(2, 3)


class BaseWAM(nn.Module):
    """Mixture-of-Transformers world-action model: pretrained Wan video branch + new action expert,
    sharing one masked attention per layer.

    Token order in the shared attention: [ cur (clean frame 0) | fut (noisy frames) | act (state + chunk) ]
    Subclasses choose whether actions may see the future (`action_sees_future`) and how to sample.
    """
    def __init__(self, dit, action_dim=7, state_dim=8, chunk=16, act_hidden=512, lam_video=1.0,
                 n_cur_frames=1, act_mlp_ratio=4, act_cross_heads=8, t_mean=0.0, t_std=1.0):
        """
        n_cur_frames     clean latent frames at the start of z (the observation); the rest are future
        act_mlp_ratio    action-expert MLP width = act_mlp_ratio * act_hidden
        act_cross_heads  heads of the action text cross-attention (must divide act_hidden)
        t_mean, t_std    logit-normal noise-level sampling: t = sigmoid(N(t_mean, t_std))
        """
        super().__init__()
        self.chunk, self.lam_video = chunk, lam_video
        self.action_dim, self.state_dim = action_dim, state_dim
        self.n_cur_frames, self.t_mean, self.t_std = n_cur_frames, t_mean, t_std
        self.grad_ckpt = False                     # set True to trade compute for activation memory

        cfg = dit.config
        self.n_heads  = cfg.num_attention_heads                            # 12 for Wan2.1-1.3B
        self.attn_dim = cfg.num_attention_heads * cfg.attention_head_dim   # 1536
        self.n_layers = cfg.num_layers                                     # 30

        self.backbone = DiTBackbone(dit)
        self.action_expert = ActionExpert(
            action_dim, state_dim, chunk, act_hidden,
            attn_dim=self.attn_dim, n_heads=self.n_heads, n_layers=self.n_layers,
            text_dim=self.attn_dim,                                        # DiT's projected text width
            mlp_ratio=act_mlp_ratio, cross_heads=act_cross_heads,
        )

        # dataset statistics for normalising actions/state (fill with set_norm_stats)
        self.register_buffer("action_mean", torch.zeros(action_dim))
        self.register_buffer("action_std",  torch.ones(action_dim))
        self.register_buffer("state_mean",  torch.zeros(state_dim))
        self.register_buffer("state_std",   torch.ones(state_dim))

    # ---- subclasses override these two ----
    def action_sees_future(self) -> bool:
        raise NotImplementedError

    @torch.no_grad()
    def sample_actions(self, z_cur, text, state, steps=10):
        raise NotImplementedError

    # ---- normalisation ----
    def set_norm_stats(self, action_mean, action_std, state_mean, state_std):
        """e.g. stats["action"]["mean"], stats["action"]["std"], stats["observation.state"][...] from load_data"""
        for name, val in [("action_mean", action_mean), ("action_std", action_std),
                          ("state_mean", state_mean), ("state_std", state_std)]:
            getattr(self, name).copy_(torch.as_tensor(val, dtype=torch.float32).flatten())

    def norm_actions(self, a):   return (a - self.action_mean) / (self.action_std + 1e-6)
    def unnorm_actions(self, a): return a * (self.action_std + 1e-6) + self.action_mean
    def norm_state(self, s):     return (s - self.state_mean) / (self.state_std + 1e-6)

    # ---- shared ----
    def build_mask(self, n_cur, n_fut, n_act, device):
        """(N, N) bool, True = row token may attend to column token."""
        N = n_cur + n_fut + n_act
        cur = slice(0, n_cur)
        fut = slice(n_cur, n_cur + n_fut)
        act = slice(n_cur + n_fut, N)

        mask = torch.zeros(N, N, dtype=torch.bool, device=device)
        mask[cur, cur] = True                       # current frame: only itself
        mask[fut, cur] = True                       # future video: present + future
        mask[fut, fut] = True
        mask[act, cur] = True                       # actions: present + actions
        mask[act, act] = True
        if self.action_sees_future():
            mask[act, fut] = True                   # imagining variants only
        return mask                                 # video -> action is never allowed

    def layer(self, i, xv, xa, timestep_proj, rope, t_emb, text_proj, text_a, mask):
        """One layer of both streams: separate pre_attn, shared masked attention, separate post_attn."""
        n_vid, n_act = xv.shape[1], xa.shape[1]
        qv, kv, vv, cache_v = self.backbone.pre_attn(i, xv, timestep_proj, rope)
        qa, ka, va, cache_a = self.action_expert.pre_attn(i, xa, t_emb)

        q = torch.cat([qv, qa.to(qv.dtype)], dim=2)       # (B, heads, n_vid + n_act, head_dim)
        k = torch.cat([kv, ka.to(kv.dtype)], dim=2)
        v = torch.cat([vv, va.to(vv.dtype)], dim=2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out_v, out_a = out.split([n_vid, n_act], dim=2)

        xv = self.backbone.post_attn(i, xv, out_v, cache_v, text_proj)
        xa = self.action_expert.post_attn(i, xa, out_a.to(xa.dtype), cache_a, text_a)
        return xv, xa

    def forward(self, z, text, noisy_actions, state, t_video, t_action):
        """
        z              (B, 16, F, H, W)  first n_cur_frames latent frames clean, the rest already noised
        text           (B, 512, 4096)    T5 embeddings
        noisy_actions  (B, chunk, 7)     normalised, already noised
        state          (B, 8)            normalised, clean
        t_video, t_action (B,)           noise levels in [0, 1]
        returns video_velocity (B, 16, F - n_cur_frames, H, W), action_velocity (B, chunk, 7)
        """
        # 1-2. embed both streams
        xv, rope, timestep_proj, temb, text_proj = self.backbone.embed(
            z, t_video, text, n_cur_frames=self.n_cur_frames)
        xa, t_emb = self.action_expert.embed(noisy_actions, state, t_action)
        text_a = text_proj.to(xa.dtype)

        # 3. mask over [cur | fut | act]
        n_vid, n_act = xv.shape[1], xa.shape[1]
        n_cur = self.n_cur_frames * self.backbone.tokens_per_frame(z.shape)
        mask = self.build_mask(n_cur, n_vid - n_cur, n_act, z.device)

        # 4. layer loop: separate pre/post, one shared attention
        for i in range(self.n_layers):
            args = (i, xv, xa, timestep_proj, rope, t_emb, text_proj, text_a, mask)
            if self.grad_ckpt and torch.is_grad_enabled():
                # recompute this layer's activations in backward instead of storing them (saves memory)
                xv, xa = torch.utils.checkpoint.checkpoint(self.layer, *args, use_reentrant=False)
            else:
                xv, xa = self.layer(*args)

        # 5. heads: drop the clean current frame(s) and the state slot
        video_velocity  = self.backbone.head(xv, temb, z.shape)[:, :, self.n_cur_frames:]
        action_velocity = self.action_expert.head(xa)
        return video_velocity, action_velocity

    # ---- KV cache for inference without future video ----
    @torch.no_grad()
    def encode_observation(self, z_cur, text):
        """Run the video branch ONCE on the clean current frame(s) and keep each layer's K/V.

        Valid because the current-frame tokens attend only to themselves (never to actions), so
        their K/V are the same at every denoising step. Returns a cache for `action_velocity`.
        """
        B = z_cur.shape[0]
        t_zero = torch.zeros(B, device=z_cur.device)
        xv, rope, timestep_proj, _, text_proj = self.backbone.embed(
            z_cur, t_zero, text, n_cur_frames=z_cur.shape[2])
        kv_cache = []
        for i in range(self.n_layers):
            qv, kv, vv, cache_v = self.backbone.pre_attn(i, xv, timestep_proj, rope)
            kv_cache.append((kv, vv))
            out_v = F.scaled_dot_product_attention(qv, kv, vv)            # cur -> cur only
            xv = self.backbone.post_attn(i, xv, out_v, cache_v, text_proj)
        return {"kv": kv_cache, "text_a": text_proj}

    def action_velocity(self, obs_cache, noisy_actions, state, t_action):
        """One denoising step using cached observation K/V: only the action expert runs.

        Action tokens attend to [cached current-frame K/V | their own K/V], exactly the keys they
        may see in training when actions cannot see the future.
        """
        xa, t_emb = self.action_expert.embed(noisy_actions, state, t_action)
        text_a = obs_cache["text_a"].to(xa.dtype)
        for i, (kv, vv) in enumerate(obs_cache["kv"]):
            qa, ka, va, cache_a = self.action_expert.pre_attn(i, xa, t_emb)
            k = torch.cat([kv, ka.to(kv.dtype)], dim=2)
            v = torch.cat([vv, va.to(vv.dtype)], dim=2)
            out_a = F.scaled_dot_product_attention(qa.to(kv.dtype), k, v)  # no mask: all keys allowed
            xa = self.action_expert.post_attn(i, xa, out_a.to(xa.dtype), cache_a, text_a)
        return self.action_expert.head(xa)

    def sample_t(self, B, device):
        """Logit-normal noise levels in (0, 1), as in Wan / Fast-WAM."""
        return torch.sigmoid(self.t_mean + self.t_std * torch.randn(B, device=device))

    def loss(self, z_clean, text, actions, state, act_pad=None):
        """
        z_clean (B, 16, F, H, W) normalised VAE latents | text (B, 512, 4096)
        actions (B, chunk, 7) raw | state (B, 8) raw | act_pad (B, chunk) bool, True = padded step
        returns (total loss, {"loss_act", "loss_video"})
        """
        assert actions.shape[1] == self.chunk, \
            f"data chunk {actions.shape[1]} != model chunk {self.chunk} (check load_suite(chunk=...))"
        assert z_clean.shape[2] > self.n_cur_frames, "z has no future frames to predict"

        B, device = z_clean.shape[0], z_clean.device
        actions, state = self.norm_actions(actions), self.norm_state(state)

        # noise the future video; the current frame(s) stay clean
        c = self.n_cur_frames
        t_video = self.sample_t(B, device)
        z_fut = z_clean[:, :, c:]
        noise_v = torch.randn_like(z_fut)
        tv = t_video.view(B, 1, 1, 1, 1).to(z_fut.dtype)
        z_noisy = torch.cat([z_clean[:, :, :c], (1 - tv) * z_fut + tv * noise_v], dim=2)

        # noise the actions
        t_action = self.sample_t(B, device)
        noise_a = torch.randn_like(actions)
        ta = t_action.view(B, 1, 1)
        noisy_actions = (1 - ta) * actions + ta * noise_a

        v_pred, a_pred = self(z_noisy, text, noisy_actions, state, t_video, t_action)

        # flow-matching targets: velocity = noise - clean
        L_vid = F.mse_loss(v_pred.float(), (noise_v - z_fut).float())
        err = (a_pred.float() - (noise_a - actions).float()).pow(2).mean(-1)   # (B, chunk)
        if act_pad is not None:
            keep = (~act_pad.bool()).float()
            L_act = (err * keep).sum() / keep.sum().clamp(min=1)
        else:
            L_act = err.mean()

        loss = L_act + self.lam_video * L_vid
        return loss, {"loss_act": L_act.detach(), "loss_video": L_vid.detach()}


class FastWAM(BaseWAM):
    """Video co-training during training; NO future imagination at test time."""
    def action_sees_future(self) -> bool:
        return False

    @torch.no_grad()
    def sample_actions(self, z_cur, text, state, steps=10, use_cache=True, noise=None):
        """
        z_cur (B, 16, 1, H, W) current-frame latent | text (B, 512, 4096) | state (B, 8) raw
        returns actions (B, chunk, 7) in robot units.

        use_cache=True : run the DiT once on the current frame, cache each layer's K/V, then run
                         only the action expert for every denoising step (Fast-WAM inference).
        use_cache=False: rerun the full model every step (slow; same result, kept for checking).
        """
        B, device = z_cur.shape[0], z_cur.device
        state = self.norm_state(state)
        a = noise.clone() if noise is not None else \
            torch.randn(B, self.chunk, self.action_dim, device=device)    # t = 1: pure noise
        ts = torch.linspace(1.0, 0.0, steps + 1, device=device)

        if use_cache:
            obs = self.encode_observation(z_cur, text)
            step = lambda a, t: self.action_velocity(obs, a, state, t)
        else:
            t_zero = torch.zeros(B, device=device)
            step = lambda a, t: self(z_cur, text, a, state, t_zero, t)[1]

        for t, t_next in zip(ts[:-1], ts[1:]):
            v = step(a, t.expand(B))
            a = a + (t_next - t) * v                                       # Euler step toward t = 0
        return self.unnorm_actions(a)

