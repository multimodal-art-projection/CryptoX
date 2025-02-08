random_port=20001


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=/root/miniconda3/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES=0,1

nohup torchrun --nproc_per_node 1 --master_port $random_port activator.py -m Qwen2.5-7B-Instruct -p Qwen/Qwen2.5-7B-Instruct 
