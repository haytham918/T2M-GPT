#!/bin/bash -l
#SBATCH --job-name=train_vq_vae
#SBATCH --output=output/VQVAE_411/eval_log.txt
#SBATCH --error=output/VQVAE_411/eval_error.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=40g
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=10:00:00
#SBATCH --account=shdpm0
##### END preamble

# module load cuda/12.3.0

source /sw/pkgs/arc/python3.10-anaconda/2023.03/etc/profile.d/conda.sh
conda activate T2M-GPT

python3 train_vq.py \
--batch-size 256 \
--lr 2e-4 \
--total-iter 300000 \
--lr-scheduler 200000 \
--nb-code 512 \
--down-t 2 \
--depth 3 \
--dilation-growth-rate 3 \
--dataname t2m \
--vq-act relu \
--quantizer ema_reset \
--loss-vel 0.5 \
--recons-loss l1_smooth \
--exp-name VQVAE \
--print-iter 25 \
--loss_ergo 0.1 \
--loss_ergo_mode same \
--note "411-same-0.1ergoloss" \
--out-dir "/scratch/shdpm_root/shdpm0/wenleyan/T2M_exp/Round2/VQVAE_411" \

# --loss_ergo_mode zero_increase \
# --note "71-zero_increase" \
# --out-dir "/scratch/shdpm_root/shdpm0/wenleyan/T2M_exp/Round2/VQVAE_71" \

# --loss_ergo_mode increase \
# --note "61-increase" \
# --out-dir "/scratch/shdpm_root/shdpm0/wenleyan/T2M_exp/Round2/VQVAE_61" \


# --loss_ergo_mode decrease \
# --note "51-decrease" \
# --out-dir "/scratch/shdpm_root/shdpm0/wenleyan/T2M_exp/Round2/VQVAE_81" \

# --loss_ergo_mode same \
# --note "41-same" \
# --out-dir "/scratch/shdpm_root/shdpm0/wenleyan/T2M_exp/Round2/VQVAE_41" \


# --eval-iter 25 \

