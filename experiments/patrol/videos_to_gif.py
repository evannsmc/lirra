"""Convert the videos of rendered patrol episodes to GIFs (palette-optimized, same frames).

    uv run --no-sync --with imageio-ffmpeg python experiments/patrol/videos_to_gif.py \
        runs/patrol_videos/x2/<run> [<run> ...] [--width 800] [--fps 8] [--views composite topdown]

For each run directory, writes ``<video>.gif`` next to each ``<system>_<spec>[_<view>].mp4`` of
the requested views (``topdown`` is the video without a view suffix).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from render_episode import to_gif


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--views", nargs="+", default=["composite", "topdown", "mujoco"],
                    choices=["composite", "topdown", "mujoco"])
    args = ap.parse_args()
    for run in args.runs:
        for mp4 in sorted(run.glob("*.mp4")):
            # <system>_<spec>[_<view>]; the spec itself may contain "_" (theta_plus)
            view = mp4.stem.rsplit("_", 1)[-1]
            view = view if view in ("composite", "mujoco") else "topdown"
            if view in args.views:
                print("wrote", to_gif(mp4, args.width, args.fps), flush=True)


if __name__ == "__main__":
    main()
