import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import ArgoverseV1Dataset


def main() -> None:
    root = "/home/lbh/HiVT/datasets/argoverse"
    for split in ("train", "val"):
        t0 = time.time()
        print(f"=== start {split} preprocessing ===", flush=True)
        dataset = ArgoverseV1Dataset(root, split)
        print(f"=== finished {split}: {len(dataset)} samples, elapsed {time.time() - t0:.1f}s ===", flush=True)
    print("=== all preprocessing done ===", flush=True)


if __name__ == "__main__":
    main()
