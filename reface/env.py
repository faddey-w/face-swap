import os


try:
    is_cloud = os.getlogin() == "ec2-user"
except OSError:
    is_cloud = True  # Colab raises error on getlogin()


if is_cloud:
    device = "cuda"
else:
    device = "cpu"


repo_root = os.path.dirname(os.path.dirname(__file__))
