#!/bin/bash 
#SBATCH --job-name=NYS_queries_on_range
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:1    
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=MCMC_logs/%x_%j.out

for i in {0..0}
do
    python mc_ontario.py --use-range --n-bursts 10000000 &
done

wait