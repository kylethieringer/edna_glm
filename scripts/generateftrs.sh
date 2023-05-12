#!/bin/bash
#SBATCH --time=0:60:00
#SBATCH --mem=64000
#SBATCH --cpus-per-task=8
#SBATCH --output='ftrs.%A.%a.log'
##SBATCH -N 1
##SBATCH --ntasks-per-socket=1
##SBATCH --ntasks-per-node=1

# you may also consider creating a separate environment specifically for creating features
module load anaconda
conda activate glm

python generateftrs.py