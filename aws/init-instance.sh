mkdir yaroslav
cd yaroslav/
git clone git@bitbucket.org:Faddey/refaceai-test.git .

echo alias l=\"ls -la \" >> /home/ec2-user/.bashrc
echo export PYTHONPATH=. >> /home/ec2-user/.bashrc
echo umask 0 >> /home/ec2-user/.bashrc
echo export LC_ALL=en_US.UTF-8 >> /home/ec2-user/.bashrc
echo export LANG=en_US.UTF-8 >> /home/ec2-user/.bashrc
echo set -g utf8 on >> /home/ec2-user/.tmux.conf
echo bind -n S-Left  previous-window >> /home/ec2-user/.tmux.conf
echo bind -n S-Right  next-window >> /home/ec2-user/.tmux.conf
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O install-conda.sh
bash install-conda.sh -b -p /home/ec2-user/miniconda3
sudo -u ec2-user /home/ec2-user/miniconda3/bin/conda init
sed -i '/anaconda/d' ~/.dlamirc
/home/ec2-user/miniconda3/bin/conda env create python=3.7 -f environment.yml
echo conda activate reface >> /home/ec2-user/.bashrc

sudo yum install -y amazon-efs-utils
sudo mkdir /mnt/efs
sudo chmod o+rw /mnt/efs
sudo mount -t efs fs-db0fa5a3:/ /mnt/efs
sudo echo "fs-db0fa5a3:/ /mnt/efs efs defaults,_netdev 0 0" | sudo tee -a /etc/fstab
ln -s /mnt/efs/models/ .models
ln -s /mnt/efs/data .data
