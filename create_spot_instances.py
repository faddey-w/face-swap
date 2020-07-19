import os
import tempfile
import json
import yaml
import shlex
import base64
import argparse


def main():

    cli = argparse.ArgumentParser()
    cli.add_argument("--n-instances", "-n", default=1, type=int)
    cli.add_argument("--instance-type", "-t", default="p3.2xlarge")
    cli.add_argument("--no-public-ip", action="store_true")
    opts = cli.parse_args()

    conda_url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"

    ssh_keys = [
        "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQDWnNkDyAK0afs6QcZoZ5+pc+/jrWaXF2gdX3gqKZ"
        "wg2STvocjsmfKRdLgwUW8DkIQPVXbwSC3xaEX/8QuiPIfOz8uhodInny/O7iixtZrP+YPd15ee2W+"
        "5iXhJGMj67lvx98UWScRbzKK/GhbF6EluV0RYSfZ2qdG0KXcsw+e+Eatvj5Dfz3XzHHx3xwqI+"
        "5f8cTLIvoYr3sUEhhMq22+OT5Jcs6373spTss2l0delj55gAPb2xLZ3360jd0z4ZcSkF0+"
        "TZ8nZ3iaKaMzo4tyz6O142tIge7MUJ2d3L+Uh/Pal9YsIiHvmiffuEmCoWBhs1kzIJGiuP4/"
        "mrRLVdIEHEtfGZJylLPFL9kM04MM9JA1x/jwgkpcOl2sczRxtsumGBUtRfH+DczUHKxglg"
        "0EI7EDKBotKuvFAqyGjXNbnHNLhE2CTs1ZKjTnRytpBEvtKz87c/+nkIEMlSZzHmXS4Bd2Z9YVNn"
        "/74bjYbAbmPUFVetQ/Foxi3oviKmmZnkoUpkL08WGaTocmmZktlgMj1gEEwBAS78RyTmb+atozo2l"
        "svmaRfQyvrgeeLAW20xzH4Mi3EdiltVm/GQonzGE6g7X/3rgiujeHhDfmgono0fhVFNRMoK5s/JLl"
        "Hoh1S+4sP3C6kfFtrXCWoUN7/qI8Y3EgCvUZKcPPD4UQtzAIuxw== yaroslav@magentiq.com"
    ]
    environment_yml = open("environment.yml").read()

    user_dir = "/home/ec2-user"
    user = "ec2-user"
    cloudinit_config = {
        "runcmd": [
            # # # # # # # # # # # # # # # # #
            # convenient stuff for shell:
            f'echo alias l=\\"ls -la \\" >> {user_dir}/.bashrc',
            f"echo export PYTHONPATH=. >> {user_dir}/.bashrc",
            f"echo umask 0 >> {user_dir}/.bashrc",
            f"echo export LC_ALL=en_US.UTF-8 >> {user_dir}/.bashrc",
            f"echo export LANG=en_US.UTF-8 >> {user_dir}/.bashrc",
            f"echo set -g utf8 on >> {user_dir}/.tmux.conf",
            f"echo bind -n S-Left  previous-window >> {user_dir}/.tmux.conf",
            f"echo bind -n S-Right  next-window >> {user_dir}/.tmux.conf",
            # # # # # # # # # # # # # # # # #
            # installing miniconda:
            # (in miniconda installation of libraries goes more smoothly)
            f"wget {conda_url} -O install-conda.sh",
            f"bash install-conda.sh -b -p {user_dir}/miniconda3",
            f"sudo -u {user} {user_dir}/miniconda3/bin/conda init",
            # # # # # # # # # # # # # # # # #
            # installing EFS:
            "yum install -y amazon-efs-utils",
            "mkdir /mnt/efs",
            "chmod o+rw /mnt/efs",
            "mount -t efs fs-db0fa5a3:/ /mnt/efs",
            'echo "fs-db0fa5a3:/ /mnt/efs efs defaults,_netdev 0 0" >> /etc/fstab',
            # # # # # # # # # # # # # # # # #
            # install our dependencies:
            f"echo {shlex.quote(environment_yml)} > environment.yml",
            f"{user_dir}/miniconda3/bin/conda env create python=3.7 -f environment.yml",
            f"chown {user}:{user} -R {user_dir}/miniconda3",
            f"echo conda activate reface >> {user_dir}/.bashrc",
        ],
        "ssh_authorized_keys": ssh_keys,
    }
    # symlinking EFS to our work directories:
    for developer in ["yaroslav"]:
        work_dir = f"{user_dir}/{developer}"
        cloudinit_config["runcmd"].extend(
            [
                f"sudo -u {user} mkdir {work_dir}",
                f"sudo -u {user} ln -s /mnt/efs/models/ {work_dir}/.models",
                f"sudo -u {user} ln -s /mnt/efs/data/ {work_dir}/.data",
            ]
        )

    userdata = "#cloud-config\n" + yaml.dump(cloudinit_config)

    if opts.no_public_ip:
        subnet_id = "subnet-78543834"
    else:
        subnet_id = "subnet-78543834"

    launch_config = {
        "ImageId": "ami-00217ec3a91d1093b",
        "InstanceType": opts.instance_type,
        "KeyName": "yaroslav",
        "IamInstanceProfile": {
            "Arn": "arn:aws:iam::180978704935:instance-profile/ec2-instance-profile"
        },
        "BlockDeviceMappings": [
            {
                "DeviceName": "/dev/xvda",
                "Ebs": {
                    "DeleteOnTermination": True,
                    "VolumeType": "gp2",
                    "VolumeSize": 90,
                    "SnapshotId": "snap-0981b271672017450",
                },
            }
        ],
        "UserData": base64.b64encode(userdata.encode("utf-8")).decode("ascii"),
        "SecurityGroupIds": [],
        "SubnetId": subnet_id,
    }
    config_str = json.dumps(launch_config)

    fd, spec_tmp_filename = tempfile.mkstemp()
    with open(spec_tmp_filename, "w") as f:
        f.write(config_str)
    os.system(
        f"aws --profile reface ec2 request-spot-instances "
        f" --instance-count {opts.n_instances} "
        f" --type persistent "
        f" --launch-specification file://{spec_tmp_filename} "
        f" --instance-interruption-behavior stop"
    )

    os.close(fd)


if __name__ == "__main__":
    main()
