FROM registry.cn-sh-01.sensecore.cn/lm4science-ccr/sense_tts_server:v1.0.0
#
# ENV https_proxy=http://proxy.sensetime.com:3128
# 防止交互式提示
ENV DEBIAN_FRONTEND=noninteractive
# 更新包列表并安装必要的软件包
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    curl \
    wget \
    vim \
    && apt-get clean

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir 

# 删除原来的/lgithllm
RUN rm -rf /lightllm
WORKDIR /lightllm
COPY . .
COPY ./resources/nltk_data  /root/nltk_data
RUN tar -xzvf /lightllm/lightllm/models/sovits_gpt/utils/text/open_jtalk_dic_utf_8-1.11.tar.gz -C /opt/conda/lib/python3.9/site-packages/pyopenjtalk

ENTRYPOINT ["python", "-m", "lightllm.server.api_server"]
# scp -r sense_tts_server root@45.77.36.225:/root/zhangxingyan/