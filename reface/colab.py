def prepare_on_colab():
    from google.colab import drive
    import os
    import sys

    drive.mount("/gdrive")
    os.symlink("/gdrive/My Drive/refaceai/data", ".data")

    sys.path.insert(0, "/gdrive/My Drive/refaceai/code")

    """
!pip install dacite facenet_pytorch dataclasses boto3
# !git clone https://github.com/NVIDIA/apex apex_repo
# !pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex_repo
    """

