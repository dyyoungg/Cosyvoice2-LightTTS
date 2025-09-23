# ARM环境

安装pip依赖

```
# docker pull ubuntu:20.04 在镜像中，在arm机器上会自动拉取架构相应的镜像
# 或者在机器上
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

# If you encounter sox compatibility issues
# ubuntu
sudo apt-get install sox libsox-dev
# centos
sudo yum install sox sox-devel
```

ttsfrd安装

```
# SDK下载
snapshot_download('iic/CosyVoice-ttsfrd', local_dir='./CosyVoice-ttsfrd')

# 或者git下载，请确保已安装git lfs
git clone https://www.modelscope.cn/iic/CosyVoice-ttsfrd.git ./CosyVoice-ttsfrd

# unzip and install
cd pretrained_models/CosyVoice-ttsfrd/
unzip resource.zip -d .
pip install ttsfrd_dependency-0.1-py3-none-any.whl
pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl
```
