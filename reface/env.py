import os


try:
    is_aws_ec2 = os.getlogin() == "ec2-user"
    is_colab = False
except OSError:
    is_colab = True  # Colab raises error on getlogin()
    is_aws_ec2 = False
is_cloud = is_colab or is_aws_ec2


if is_aws_ec2:
    device = "cuda"
elif is_colab:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = "cpu"


repo_root = os.path.dirname(os.path.dirname(__file__))
