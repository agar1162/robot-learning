"""Plot a train.py log.csv as an SVG (no matplotlib needed). Use --watch to keep it updating.

    python plot_log.py runs/overfit/log.csv              # writes runs/overfit/loss_plot.svg once
    python plot_log.py runs/overfit/log.csv --watch 10   # rewrite every 10 s
"""
import argparse
import csv
import time
from pathlib import Path

W, H, PAD_L, PAD_R, PAD_T, PAD_B = 640, 210, 56, 16, 30, 28
COLORS = {"train": "#7F77DD", "smooth": "#3C3489", "val": "#D85A30", "train_eval": "#1D9E75",
          "baseline": "#888780"}


def read(path):
    rows = list(csv.DictReader(open(path)))
    def col(name):
        return [(int(r["step"]), float(r[name])) for r in rows if r.get(name) not in (None, "")]
    return {k: col(k) for k in ["loss", "loss_act", "loss_video", "val_loss_act", "val_loss_video",
                                "val_action_l1", "sec_per_step", "train_loss_act", "train_action_l1",
                                "baseline_action_l1"]}


def ema(points, a=0.8):
    out, s = [], None
    for x, y in points:
        s = y if s is None else a * s + (1 - a) * y
        out.append((x, s))
    return out


def panel(title, series, y0, max_step):
    """series: list of (label, points, color, style) -> SVG <g> at vertical offset y0."""
    pts = [p for _, ps, _, _ in series for p in ps]
    g = [f'<text x="{PAD_L}" y="{y0 + 18}" font-size="13" font-weight="500" fill="#333">{title}</text>']
    if not pts:
        return g + [f'<text x="{PAD_L}" y="{y0 + 60}" font-size="12" fill="#888">waiting for data...</text>']
    ys = [y for _, y in pts]
    lo, hi = min(0.0, min(ys)), max(ys) * 1.05 or 1.0
    pw, ph = W - PAD_L - PAD_R, H - PAD_T - PAD_B
    sx = lambda x: PAD_L + pw * x / max(1, max_step)
    sy = lambda y: y0 + PAD_T + ph * (1 - (y - lo) / (hi - lo))
    for i in range(5):                                      # gridlines + y labels
        v = lo + (hi - lo) * i / 4
        g.append(f'<line x1="{PAD_L}" x2="{W - PAD_R}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="#eee"/>')
        g.append(f'<text x="{PAD_L - 6}" y="{sy(v) + 4:.1f}" font-size="10" fill="#888" '
                 f'text-anchor="end">{v:.3g}</text>')
    for i in range(6):                                      # x labels
        s = max_step * i / 5
        g.append(f'<text x="{sx(s):.1f}" y="{y0 + H - 8}" font-size="10" fill="#888" '
                 f'text-anchor="middle">{int(s)}</text>')
    lx = W - PAD_R
    for label, ps, color, style in series:
        if not ps:
            continue
        if style == "dots":
            g += [f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.5" fill="{color}"/>' for x, y in ps]
        else:
            d = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in ps)
            op, wd = (0.35, 1) if style == "faint" else (1, 2)
            g.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="{wd}" opacity="{op}"/>')
        if style != "faint":
            g.append(f'<text x="{lx}" y="{y0 + 18}" font-size="11" fill="{color}" text-anchor="end">'
                     f'{label}: {ps[-1][1]:.4f}</text>')
            lx -= 7 * len(f"{label}: {ps[-1][1]:.4f}") + 16
    return g


def render(log_path, out_path):
    d = read(log_path)
    max_step = max([x for v in d.values() for x, _ in v] + [1])
    speed = f" · {d['sec_per_step'][-1][1]:.2f}s/step" if d["sec_per_step"] else ""
    body = [f'<text x="{PAD_L}" y="20" font-size="14" fill="#111">{log_path} · step {max_step}{speed} '
            f'· updated {time.strftime("%H:%M:%S")}</text>']
    panels = [
        ("Action loss", [("train", d["loss_act"], COLORS["train"], "faint"),
                         ("train (smoothed)", ema(d["loss_act"]), COLORS["smooth"], "line"),
                         ("eval train", d["train_loss_act"], COLORS["train_eval"], "dots"),
                         ("eval held-out", d["val_loss_act"], COLORS["val"], "dots")]),
        ("Video loss", [("train", d["loss_video"], COLORS["train"], "faint"),
                        ("train (smoothed)", ema(d["loss_video"]), COLORS["smooth"], "line"),
                        ("eval held-out", d["val_loss_video"], COLORS["val"], "dots")]),
        ("Sampled action error (L1, robot units) - lower is better",
         [("mean-action baseline", d["baseline_action_l1"], COLORS["baseline"], "line"),
          ("train eps", d["train_action_l1"], COLORS["train_eval"], "line"),
          ("held-out eps", d["val_action_l1"], COLORS["val"], "line")]),
    ]
    for i, (title, series) in enumerate(panels):
        body += panel(title, series, 30 + i * H, max_step)
    total_h = 30 + len(panels) * H + 10
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{total_h}" '
           f'font-family="sans-serif"><rect width="100%" height="100%" fill="white"/>'
           + "".join(body) + "</svg>")
    tmp = Path(out_path).with_suffix(".tmp")
    tmp.write_text(svg)
    tmp.replace(out_path)                                   # atomic: viewers never see a half file


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out", default=None, help="default: loss_plot.svg next to the log")
    ap.add_argument("--watch", type=float, default=0, help="seconds between redraws; 0 = once")
    a = ap.parse_args()
    out = a.out or str(Path(a.log).with_name("loss_plot.svg"))
    while True:
        if Path(a.log).exists():
            render(a.log, out)
        if not a.watch:
            break
        time.sleep(a.watch)
