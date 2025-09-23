export PYTHONPATH="/mnt/afs/yangdeyu/dependency/lightllm-cosyvoice-old/lightllm-cosyvoice:$PYTHONPATH"

python -m light_tts.server.api_server \
    --model_dir /mnt/afs/share/CosyVoice2-0.5B \
    --host 0.0.0.0 \
    --port 8089 \
    --bert_process_num 1 \
    --decode_process_num 1 \
    --max_total_token_num 60000 \
    --encode_paral_num 50 \
    --gpt_paral_num 50 \
    --decode_paral_num 1 \
    --mode triton_flashdecoding \
    --log_path_or_dir "/mnt/afs/yangdeyu/dependency/lightllm-cosyvoice-old/lightllm-cosyvoice/logs/cosyvoice.log"
