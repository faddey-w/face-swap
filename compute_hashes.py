import argparse
import os
import csv
import hashlib
import tqdm
from face_swap.utils import map_unordered_fast, prefetch


def main():
    root_paths = [".data/train", ".data/test"]
    ckpt_path = ".local/checkpoints/data-hashes.csv"

    def get_hash(path, chunksize=4096 * 1024):
        md5 = hashlib.md5()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(chunksize)
                if not chunk:
                    break
                md5.update(chunk)
        return path, md5.hexdigest()

    paths_gen = (
        os.path.join(parent, file)
        for root in root_paths
        for parent, dirs, files in os.walk(root)
        for file in files
    )

    try:
        with open(ckpt_path) as f:
            ckpt = dict(csv.reader(f))
        paths_gen = (path for path in paths_gen if path not in ckpt)
    except FileNotFoundError:
        pass

    with open(ckpt_path, "a") as f:
        ckpt_wrt = csv.writer(f)
        for path, hash_ in tqdm.tqdm(
            map_unordered_fast(get_hash, paths_gen, num_workers=8, buffer_size=1000)
        ):
            ckpt_wrt.writerow([path, hash_])


if __name__ == "__main__":
    main()
