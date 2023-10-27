#!/bin/bash
#SBATCH -J he #Slurm job name

# Choose partition (queue) "gpu" or "gpu-share"
#SBATCH -p gpu
#SBATCH --output=he.out
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00 --mem=246000M

# Go to the job submission directory and run your application

cd /ceph/hpc/home/euliz/LiZhou/FmLAMA-master
source activate torch_tacred
# Execute applications in parallel

python access_LMs/run_prompting_experiment.py --lang he --dataset_dir /ceph/hpc/home/euliz/LiZhou/FmLAMA-master/data_filter/ --templates_file_path "/ceph/hpc/home/euliz/LiZhou/FmLAMA-master/data_filter/he/templates.jsonl" --device cuda --rel country_1 country_2 country_3 country_4 country_5 hasParts_1 hasParts_2 hasParts_3 hasParts_4 hasParts_5 country_2-1 country_2-2 country_2-3 country_2-4 country_2-5 country_3-1 country_3-2 country_3-3 country_3-4 country_3-5 country_4-1 country_4-2 country_4-3 country_4-4 country_4-5
