def prepare_on_colab():
    from google.colab import drive
    import tarfile
    import tqdm
    import os

    drive.mount("/gdrive")

    paths = [
        [
            "/gdrive/My Drive/refaceai/bb_landmark.tar.gz",
            "/gdrive/My Drive/refaceai/bb_landmark",
        ],
        [
            "/gdrive/My Drive/refaceai/vggface2_test.tar.gz",
            "/gdrive/My Drive/refaceai/test",
        ],
        [
            "/gdrive/My Drive/refaceai/vggface2_train.tar.gz",
            "/gdrive/My Drive/refaceai/train",
        ],
    ]

    for archive, tgt_dir in paths:
        os.makedirs(tgt_dir, exist_ok=True)
        with tarfile.open(archive, "r:gz") as tarf:
            tarf.extractall(tgt_dir, tqdm.tqdm(tarf.getmembers(), desc=tgt_dir))
