#!/usr/bin/env bash
# SBATCH --job-name=n-gram
# SBATCH --partition=aa100
# SBATCH --qos=normal
# SBATCH --gres=gpu:1
# SBATCH --nodes=1
# SBATCH --ntasks=1
# SBATCH --cpus-per-task=16
# SBATCH --mem=120G
# SBATCH --time=2:00:00
# SBATCH --output=n_gram.log
# SBATCH --output=n_gram.err
# SBATCH --mail-type=END,FAIL
