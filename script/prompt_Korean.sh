#!/bin/bash
#SBATCH -J ko #Slurm job name

# Choose partition (queue) "gpu" or "gpu-share"
#SBATCH -p gpu
#SBATCH --output=ko.out
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00 --mem=246000M

# Go to the job submission directory and run your application

cd /ceph/hpc/home/euliz/LiZhou/FmLAMA-master
source activate torch_tacred
# Execute applications in parallel

python access_LMs/run_prompting_experiment.py --lang ko --dataset_dir /ceph/hpc/home/euliz/LiZhou/FmLAMA-master/data_filter/ --templates_file_path "/ceph/hpc/home/euliz/LiZhou/FmLAMA-master/data_filter/ko/templates.jsonl" --device cuda --rel country_1 country_2 country_3 country_3-2 country_3-3 country_3-4 country_4 country_4-2 country_5 country_5-2 hasParts_1 hasParts_2 hasParts_2-2 hasParts_2-3 hasParts_2-4 hasParts_3 hasParts_3-2 hasParts_3-3 hasParts_3-4 hasParts_4 hasParts_4-2 hasParts_5
