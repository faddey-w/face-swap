import os
import argparse
import tqdm
import botocore.exceptions
from reface.utils import map_unordered_fast, prefetch
from reface import s3_utils


def iter_files(split_name):
    for parent, dirs, files in os.walk(os.path.join(".data", split_name)):
        for file in files:
            yield os.path.join(parent, file)


def check_if_not_uploaded(local_path):
    try:
        s3_utils.get_s3_client().head_object(
            Bucket=s3_utils.bucket, Key=local_path_to_s3_key(local_path)
        )
    except botocore.exceptions.ClientError:
        return True
    else:
        return False


def upload(local_path, check):
    if not check or check_if_not_uploaded(local_path):
        s3_utils.get_s3_client().upload_file(
            Bucket=s3_utils.bucket,
            Key=local_path_to_s3_key(local_path),
            Filename=local_path,
        )
    return local_path


def local_path_to_s3_key(local_path):
    rel_path = os.path.relpath(local_path, ".data")
    return rel_path


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("split", choices=["train", "test", "bb_landmark"])
    cli.add_argument("start_after", nargs="?")
    cli.add_argument("--check", action="store_true")
    cli.add_argument("--workers", type=int)
    opts = cli.parse_args()
    checkpoint_filename = f"upload-checkpoint-{opts.split}.txt"

    files_gen = tqdm.tqdm(iter_files(opts.split), desc="listing files", leave=False)
    if os.path.exists(checkpoint_filename):
        with open(checkpoint_filename) as f:
            already_done = set(line.strip() for line in f)
        not_done_files = (f for f in files_gen if f not in already_done)
    else:
        not_done_files = (f for f in files_gen)
    not_done_files = tqdm.tqdm(not_done_files, desc="files to upload", leave=False)

    def iter_not_done_files_and_set_total():
        total = 0
        for total, item in enumerate(not_done_files, 1):
            yield item
        try:
            result.total = total
        except AttributeError:
            pass

    result = tqdm.tqdm(
        map_unordered_fast(
            lambda p: upload(p, check=opts.check),
            prefetch(iter_not_done_files_and_set_total()),
            num_workers=opts.workers,
            buffer_size=1000,
        ),
        desc="done",
        smoothing=0,
    )

    with open(checkpoint_filename, "a") as f:
        for path in result:
            f.write(path + "\n")


if __name__ == "__main__":
    main()
