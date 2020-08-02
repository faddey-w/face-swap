import os
import csv
import tqdm
from concurrent.futures.thread import ThreadPoolExecutor
from queue import Queue
from face_swap import s3_utils


def main():
    dirs_q = Queue(1000)
    outs_q = Queue(1000)

    def list_dir_and_enqueue():
        prefix = dirs_q.get_nowait()
        params = dict(Bucket=s3_utils.bucket, Delimiter="/")
        if prefix:
            params["Prefix"] = prefix

        while True:

            response = s3_utils.get_s3_client().list_objects(**params)

            for item in response.get("Contents", []):
                outs_q.put(item)
            for item in response.get("CommonPrefixes", []):
                dirs_q.put(item["Prefix"])
                executor.submit(list_dir_and_enqueue)

            if response["IsTruncated"]:
                params["Marker"] = response["NextMarker"]
            else:
                break
        dirs_q.task_done()

    executor = ThreadPoolExecutor(20)

    dirs_q.put("test")
    executor.submit(list_dir_and_enqueue)

    def wait_and_put_end():
        dirs_q.join()
        outs_q.put(None)

    executor.submit(wait_and_put_end)

    with open(".local/checkpoints/data-hashes.csv") as f:
        local_hashes = {p: h for p, h in csv.reader(f)}

    pbar = tqdm.tqdm(desc="verifying S3")
    n_differences = 0

    f_train = open(".local/checkpoints/upload-checkpoint-train.txt", "w")
    f_test = open(".local/checkpoints/upload-checkpoint-test.txt", "w")
    with f_train, f_test:
        while True:
            item = outs_q.get()
            if item is None:
                break
            path = os.path.join(".data", item["Key"])
            remote_hash = item["ETag"]

            if path not in local_hashes:
                continue

            # import pdb; pdb.set_trace()
            if remote_hash[1:-1] == local_hashes[path]:
                if "/train/" in path:
                    f_train.write(path + "\n")
                else:
                    f_test.write(path + "\n")
            else:
                n_differences += 1
                pbar.desc = f"verifying S3 (diff={n_differences})"
            pbar.update()

    executor.shutdown()


if __name__ == "__main__":
    main()
