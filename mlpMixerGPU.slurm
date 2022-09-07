#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=MlpMixer          #Set the job name to "MlpMixer"
#SBATCH --time=03:30:00              #Set the wall clock limit to 3hr and 30min
#SBATCH --nodes=2                    #Number of nodes 
#SBATCH --ntasks-per-node=2          #Request 2 task
#SBATCH --cpus-per-task=8            #Number of CPUs per task
#SBATCH --mem=26GB                   #Request 20GB (20GB) per node
#SBATCH --error=MlpMixer.%J.err_     #send stdout/err to "[jobID].err"
#SBATCH --output=MlpMixer.%J.out_    #Send stdout/out to "[jobID].out"
#SBATCH --gres=gpu:2                 #Request 2 GPU per node
#SBATCH --partition=lyceum           #Request the GPU partition/queue

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=12340
export WORLD_SIZE=4

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

eval "$(conda shell.bash hook)"
conda activate mlenv

cd $HOME/MLPMixer

#seeds = [42, 27, 14, 924, 80]

#srun python main.py --name daphnet_step10_window90_patch9_downsample2_lr0_01_BIG --dataset daphnet --no-verbose --no-pretrain --step_size 10 --window_size 90 --num_workers 4 --saving_folder saved/daphnet_adam --save_best_metric macro_f1 --balance_batch --nb_steps 10000 --downsample_factor 2 --lr 0.01 --mlp_patch_size 9 --mlp_num_blocks 10 --mlp_token_dim 384 --mlp_patch_dim 768 --mlp_channel_dim 3072

#srun python main.py --name pamap2_step10_window77_patch11_downsample3_lr0_01_BIG --dataset pamap2 --no-verbose --no-pretrain --step_size 10 --window_size 77 --num_workers 4 --saving_folder saved/pamap2_adam --save_best_metric macro_f1 --balance_batch --nb_steps 10000 --downsample_factor 3 --lr 0.01 --mlp_patch_size 11 --mlp_num_blocks 10 --mlp_token_dim 384 --mlp_patch_dim 768 --mlp_channel_dim 3072

#srun python main.py --name oppo_gestures_step11_window77_patch11_downsample3_lr0_01_BIG --dataset opportunity_gestures --no-verbose --no-pretrain --step_size 11 --window_size 77 --num_workers 4 --saving_folder saved/oppo_weighted_adam --save_best_metric weighted_f1 --balance_batch --nb_steps 10000 --lr 0.01 --mlp_patch_size 11 --mlp_num_blocks 10 --mlp_token_dim 384 --mlp_patch_dim 768 --mlp_channel_dim 3072

#srun python main.py --name oppo_locomotion_step11_window77_patch11_downsample3_lr0_01_BIG --dataset opportunity_locomotion --no-verbose --no-pretrain --step_size 11 --window_size 77 --num_workers 4 --saving_folder saved/oppo_weighted_adam --save_best_metric weighted_f1 --balance_batch --nb_steps 10000 --lr 0.01 --mlp_patch_size 11 --mlp_num_blocks 10 --mlp_token_dim 384 --mlp_patch_dim 768 --mlp_channel_dim 3072

#srun python main.py --name pamap2_step10_window80_52_patch13_downsample3_lr0_01 --dataset pamap2 --no-verbose --no-pretrain --step_size 10 --window_size 80 --num_workers 4 --saving_folder saved/pamap2_adam --save_best_metric macro_f1 --balance_batch --nb_steps 10000 --downsample_factor 3 --lr 0.01 --mlp_patch_size 4

#srun python main.py --name oppo_locomotion_step3_window99_patch11_downsample3_lr0_01_s42 --dataset opportunity_locomotion --no-verbose --no-pretrain --step_size 3 --window_size 99 --num_workers 4 --saving_folder saved/oppo --save_best_metric weighted_f1 --balance_batch --nb_steps 10000 --lr 0.01 --mlp_patch_size 11 --seed 42

#srun python main.py --name oppo_gestures_step3_window77_patch11_downsample3_lr0_01_s42_NoToken --dataset opportunity_gestures --no-verbose --no-pretrain --step_size 3 --window_size 77 --num_workers 4 --saving_folder saved/oppo --save_best_metric weighted_f1 --balance_batch --nb_steps 10000 --lr 0.01 --mlp_patch_size 11 --seed 42  --weight_decay 1e-3 --mlp_no_token

srun python main.py --name pamap2_step3_window84_patch4_downsample3_lr0_01_s42_NoChannel --dataset pamap2 --no-verbose --no-pretrain --step_size 3 --window_size 84 --num_workers 4 --saving_folder saved/pamap2 --save_best_metric macro_f1 --balance_batch --nb_steps 10000 --downsample_factor 3 --lr 0.01 --mlp_patch_size 4 --mlp_num_blocks 10 --mlp_no_channel

#srun python main.py --name daphnet_step3_window126_patch9_downsample2_lr0_01_s42_NoRGB_4 --dataset daphnet --no-verbose --no-pretrain --step_size 3 --window_size 126 --num_workers 4 --saving_folder saved/daphnet --save_best_metric macro_f1 --balance_batch --nb_steps 10000 --downsample_factor 2 --lr 0.01 --mlp_patch_size 9  --seed 42 --mlp_channel_dim 512 --mlp_no_RGB_embed

#srun python main.py --name daphnet_step3_window90_patch9_downsample2_lr0_01_SMALL_num15_s42 --dataset daphnet --no-verbose --no-pretrain --step_size 3 --window_size 90 --num_workers 4 --saving_folder saved/daphnet --save_best_metric macro_f1 --balyance_batch --nb_steps 10000 --downsample_factor 2 --lr 0.01 --mlp_patch_size 9  --seed 42 --mlp_num_blocks 15 --mlp_token_dim 128 --mlp_patch_dim 256 --mlp_channel_dim 512