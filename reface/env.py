import os


try:
    is_aws_ec2 = os.getlogin() == "ec2-user"
    is_colab = False
except OSError:
    is_colab = True  # Colab raises error on getlogin()
    is_aws_ec2 = False
is_cloud = is_colab or is_aws_ec2


if is_cloud:
    device = "cuda"
else:
    device = "cpu"


repo_root = os.path.dirname(os.path.dirname(__file__))
