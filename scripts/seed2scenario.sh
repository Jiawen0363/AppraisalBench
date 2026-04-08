#!/bin/bash
#SBATCH --job-name=seed2scenario_200
#SBATCH --partition=gpu
#SBATCH --account=你的账号名
#SBATCH --output=/home/jiawen/AppraisalBench/output/seed2scenario/slurm_%j.out
#SBATCH --error=/home/jiawen/AppraisalBench/output/seed2scenario/slurm_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail
cd /home/jiawen/AppraisalBench
source /home/jiawen/miniconda3/etc/profile.d/conda.sh
conda run -n appbench python seed2scenario/run_seed2scenario.py \
  --limit 200 \
  --output /home/jiawen/AppraisalBench/output/seed2scenario/scenarios_200.jsonl