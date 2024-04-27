#!/bin/bash
#SBATCH --job-name=qa_l13
#SBATCH -p gpu --gres=gpu:1
#SBATCH --time=23:30:00
#SBATCH --output=logs/qa_l13.out
#SBATCH --account=d2023d06-049-users
#SBATCH --ntasks=1 --cpus-per-task=8 --mem=30GB

echo $SLURMD_NODENAME $CUDA_VISIBLE_DEVICES

. /etc/profile.d/modules.sh
eval "$(conda shell.bash hook)"
conda activate /ceph/hpc/home/eujongc/anaconda3/envs/latentops
nvidia-smi

huggingface-cli login --token=your_token
echo Y

# model_map = {"vicuna-7b-v1.5": "lmsys/vicuna-7b-v1.5",
#         "vicuna-13b-v1.5": "lmsys/vicuna-13b-v1.5",
#         "Llama-2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
#         "Llama-2-13b-chat-hf": "meta-llama/Llama-2-13b-chat-hf"}

model_name="Llama-2-13b-chat-hf"
task="qa" # qa, wc


if [ $task = "qa" ]
then
    python QA_prompt.py \
        --model $model_name 
elif [ $task = "wc" ]
then
    python WC_prompt.py \
        --model $model_name 
fi