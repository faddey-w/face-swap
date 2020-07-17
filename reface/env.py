import os


try:
    os.getlogin()
except OSError:
    is_colab = True
else:
    is_colab = False


if is_colab:
    device = "cuda"
else:
    device = "cpu"
