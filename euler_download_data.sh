#!/bin/bash -l
#SBATCH --job-name=download_data
#SBATCH --output=/cluster/project/cvg/students/shangwu/FrontierNet/log/job_%j.out
#SBATCH --error=/cluster/project/cvg/students/shangwu/FrontierNet/log/job_%j.err
#SBATCH --mem-per-cpu=16G
#SBATCH --ntasks=1
#SBATCH --time=4:00:00
#SBATCH --account ls_polle

module load stack/2024-06
module load eth_proxy

cd $SCRATCH

echo "Downloading data..."
gdown 14o-rFfalQ6HvWpP27Qw-xnuQp80_G_x1
echo "Unzipping data..."
unzip -q -j Actmap_v3.zip "Actmap_v3/image/*" -d $SCRATCH/image/