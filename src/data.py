import random
from pathlib import Path
from typing import Literal, Sequence
from torch.utils.data import ConcatDataset, DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from torchvision.transforms import v2

DATA_ROOT = Path("/scratch/agarcia05/data/LIBERO-fastwam/lerobot_v30")
Suite = Literal["libero_spatial", "libero_object", "libero_goal", "libero_10"]
Split = Literal["all", "train", "val"]
SUITES: tuple[Suite, ...] = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SUITE_DIRS = {
    "libero_spatial": "libero_spatial_no_noops_lerobot",
    "libero_object":  "libero_object_no_noops_lerobot",
    "libero_goal":    "libero_goal_no_noops_lerobot",
    "libero_10":      "libero_10_no_noops_lerobot",
}


def load_suite(suite: Suite, chunk: int = 16, n_future: int = 16,
               split: Split = "all", val_frac: float = 0.05,
               seed: int = 0, size: int = 224, max_episodes: int | None = None) -> LeRobotDataset:
    """max_episodes: keep only this many (randomly chosen) episodes of the split, for small runs.
    The train/val split is decided first, so val episodes never leak into a train subset."""

    if suite not in SUITES:
        raise ValueError(f"Invalid suite {suite!r}; must be one of {SUITES}")
    if split not in ("all", "train", "val"):
        raise ValueError(f"Invalid split {split!r}")
    root = DATA_ROOT / SUITE_DIRS[suite]
    if not (root / "meta" / "info.json").exists():
        raise FileNotFoundError(f"No LeRobot dataset at {root}")

    meta = LeRobotDatasetMetadata(repo_id=suite, root=root)
    fps = meta.fps
    cams = [k for k in meta.features if k.startswith("observation.images")]
    delta = {"action": [i / fps for i in range(chunk)]}
    delta |= {cam: [i / fps for i in range(n_future + 1)] for cam in cams}

    episodes = None
    if split != "all":
        eps = list(range(meta.total_episodes))
        random.Random(seed).shuffle(eps)
        n_val = max(1, int(len(eps) * val_frac))
        chosen = eps[:n_val] if split == "val" else eps[n_val:]
        episodes = sorted(chosen[:max_episodes] if max_episodes else chosen)
    elif max_episodes:
        eps = list(range(meta.total_episodes))
        random.Random(seed).shuffle(eps)
        episodes = sorted(eps[:max_episodes])

    return LeRobotDataset(
        repo_id=suite, root=root, delta_timestamps=delta, episodes=episodes,
        video_backend="torchcodec",
        image_transforms=v2.Resize((size, size), antialias=True),
    )


def load_data(suites: Sequence[Suite] = SUITES, **kw):
    datasets = [load_suite(s, **kw) for s in suites]
    stats = {s: d.meta.stats for s, d in zip(suites, datasets)}
    ds = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    return ds, stats


def make_loader(ds, batch_size: int = 8, shuffle: bool = True, num_workers: int = 8,
                generator=None) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=True, persistent_workers=num_workers > 0, drop_last=shuffle,
                      generator=generator)