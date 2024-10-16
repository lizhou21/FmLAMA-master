#!/bin/bash
#SBATCH -J gpt4o #Slurm job name

# Choose partition (queue) "gpu" or "gpu-share"
#SBATCH --partition=p-cpu
#SBATCH -A t00120220002
#SBATCH --output=gpt4o.out
#SBATCH -N 1



# Go to the job submission directory and run your application
module load cuda11.8/toolkit/11.8.0
module load cuda12.1/toolkit/12.1.1

source /cm/shared/apps/anaconda3/etc/profile.d/conda.sh

cd /mntcephfs/lab_data/zhouli/personal/FmLAMA/analysis/04-gpt4o-evaluation
conda activate czh
# Execute applications in parallel

python gpt4o-predict.py