#!/bin/bash
#SBATCH -J ar #Slurm job name

# Choose partition (queue) "gpu" or "gpu-share"
#SBATCH -p gpu
#SBATCH --output=ar.out
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00 --mem=246000M

# Go to the job submission directory and run your application

cd /ceph/hpc/home/euliz/LiZhou/FmLAMA-master
source activate torch_tacred
# Execute applications in parallel

python access_LMs/run_prompting_experiment.py --lang ar --dataset_dir /ceph/hpc/home/euliz/LiZhou/FmLAMA-master/data_filter/ --templates_file_path "/ceph/hpc/home/euliz/LiZhou/FmLAMA-master/data_filter/ar/templates.jsonl" --device cuda --rel country_1 country_2 country_3 country_4 country_5 hasParts_1 hasParts_2 hasParts_3 hasParts_4 hasParts_5
