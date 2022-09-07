# MLPMixerHAR

MLP Mixer For Human Activity Recognition - Pytorch

### Model
The MLP-Mixer is in ./models/
### To use the pretrain Mixer download Pre-trained model (Google's Official Checkpoint)
* [Available models](https://console.cloud.google.com/storage/browser/mixer_models): Mixer-B_16, Mixer-L_16
  * imagenet-21k pre-train models
    * Mixer-B_16, Mixer-L_16
```
# Imagenet-21k pre-train
wget https://storage.googleapis.com/mixer_models/imagenet21k/{MODEL_NAME}.npz
```

### Datasets
The dataset managers are in the folder ./datasets/
The datasets used are:

Opportunity
https://archive.ics.uci.edu/ml/datasets/opportunity+activity+recognition

Daphner Gait 
https://archive.ics.uci.edu/ml/datasets/Daphnet+Freezing+of+Gait

PAMAP2
https://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring


### Usage Examples
```
python3 main.py --name pamap2_Mixer --dataset pamap2 --no-verbose --no-pretrain --step_size 3 --window_size 84 --num_workers 4 --saving_folder saved/pamap2 --save_best_metric macro_f1 --balance_batch --nb_steps 10000 --downsample_factor 3 --lr 0.01 --mlp_patch_size 4 --mlp_num_blocks 10

python2 main.py --name oppo_locomotion_Mixer --dataset opportunity_locomotion --no-verbose --no-pretrain --step_size 3 --window_size 99 --num_workers 4 --saving_folder saved/oppo --save_best_metric weighted_f1 --balance_batch --nb_steps 10000 --lr 0.01 --mlp_patch_size 11 --seed 42

python2 main.py --name daphnet_Mixer --dataset daphnet --no-verbose --no-pretrain --step_size 3 --window_size 126 --num_workers 4 --saving_folder saved/daphnet --save_best_metric macro_f1 --balance_batch --nb_steps 10000 --downsample_factor 2 --lr 0.01 --mlp_patch_size 9  --seed 42 --mlp_channel_dim 512 --mlp_no_RGB_embed
```
