random_port=20001
model_size=Qwen2.5-7B-Instruct
model_path=/map-vepfs/models/jiajun/Qwen/Qwen2.5-7B-Instruct

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=/root/miniconda3/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0,1

torchrun --nproc_per_node 1 --master_port $random_port logit_extractor.py -m $model_size -p $model_path

python explainer.py -m $model_size