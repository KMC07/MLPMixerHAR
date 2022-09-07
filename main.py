import argparse
import builtins
import json
import logging
import os
import sys
import pandas as pd
import numpy as np
import torch
import utils
from datetime import timedelta
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler, WeightedRandomSampler
from data.sampler import  BalancedBatchSampler, DistributedSamplerWrapper
from torch.autograd import Variable
from torchvision import transforms, datasets
from models import MLPMixer, train, test
from models.configs import CONFIGS
from models.train import set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True,
                    help="Name of this run. Used for monitoring.")
parser.add_argument('-d', '--dataset', choices=["opportunity_gestures", "opportunity_locomotion", "pamap2", "daphnet"],
                    type=str, default='opportunity_locomotion',
                    help="The dataset to use.")
parser.add_argument('--validation', action=argparse.BooleanOptionalAction, default=True,
                    help="Whether to use split the training set into training and validation sets.")
parser.add_argument('--downsample_factor',type=int,default=1)
parser.add_argument('--sliding_window', action=argparse.BooleanOptionalAction, default=True,
                    help="Should sliding window be used on the dataser")
parser.add_argument('--window_size', type=int, default=113,
                    help="The length of the sliding window.")
parser.add_argument('--step_size', type=int, default=3,
                    help="The stride of the sliding window.")

parser.add_argument('--saving_folder', type=str, default='saved/oppo_balanced',
                    help="The directory where the weights, history and runtime info is saved.")
parser.add_argument('--log_saving_directory', type=str, default='logs',
                    help="The directory in the saving folder where the logs are saved.")
parser.add_argument('--summary_saving_directory', type=str, default='runs',
                    help="The directory in the saving folder where the runs(model history, tensorboard, etc) are saved.")
parser.add_argument('--save_best_metric', choices=["accuracy", "micro_f1", "macro_f1", "weighted_f1"],
                    default="weighted_f1",
                    help="Metric to determine when to save the model.")
parser.add_argument('--save_final_model', action=argparse.BooleanOptionalAction, default=True,
                    help="Whether to save the final model as well.")
parser.add_argument('--train_batch_size', type=int, default=64,
                    help="The size of the batches in the training dataloaders.")
parser.add_argument("--eval_batch_size", type=int, default=64,
                    help="Total batch size for eval.")
parser.add_argument("--eval_every", type=int, default=100,
                    help="Run prediction on validation set every x steps."
                         "Will always run one evaluation at the end of training.")
parser.add_argument('--num_workers', type=int,default=0,
                    help="The number of workers to in the dataloaders.")

parser.add_argument('--balance_batch', action=argparse.BooleanOptionalAction, default=True,
                    help="Should use the balanced batch loader to balance each batch")
parser.add_argument('--lr',type=float,default=3e-2,
                    help="The initial learning rate.")
parser.add_argument('--nb_steps' , type=int,default=10000,
                    help="The number of training steps to perform.")
parser.add_argument('--weight_decay', type=float,default=0,
                    help="The weight decay if it is applied.")
parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                    help="How to decay the learning rate.")
parser.add_argument("--warmup_steps", default=500, type=int,
                    help="Step of training to perform learning rate warmup for.")
parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")
parser.add_argument('--momentum',type=float,default=0.9,
                    help="Momentum of the optimizer.")
parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=True)

parser.add_argument('--mlp_patch_size', type=int, default=8,
                    help="This is the patch size of the mlp mixer.")
parser.add_argument('--mlp_patch_dim', type=int, default=512,
                    help="This is the path dimensiont of the mlp mixer.")
parser.add_argument('--mlp_token_dim', type=int, default=256,
                    help="This is the token dimensions of the mlp mixer.")
parser.add_argument('--mlp_channel_dim', type=int, default=2048,
                    help="This is the channel dimensions of the mlp mixer.")
parser.add_argument('--mlp_num_blocks', type=int, default=10,
                    help="This is the number of blocks/layers in the mlp mixer")
parser.add_argument('--mlp_no_token', action=argparse.BooleanOptionalAction, default=False,
                    help="remove the token-mixing layer for ablation study")
parser.add_argument('--mlp_no_channel', action=argparse.BooleanOptionalAction, default=False,
                    help="remove the channel-mixing layer for ablation study")
parser.add_argument('--mlp_no_RGB_embed', action=argparse.BooleanOptionalAction, default=False,
                    help="remove the RGB Embedding layer for ablation study")
                    

parser.add_argument('--pretrain', action=argparse.BooleanOptionalAction, default=False,
                    help="Whether the model should be pretrained or trained from scratch")
parser.add_argument('--model_type', choices=["Mixer-B_16", "Mixer-L_16"],
                    default="Mixer-B_16",
                    help="Which model to use.")
parser.add_argument('-pretrain_weights','--pretrained_file', type=str, default='./weights/imagenet21k_Mixer-B_16.npz',
                    help="The directory where the pretrained weights are.")

parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument("--local_rank", type=int, default=-1,
                    help="local_rank for distributed training on gpus")
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--dist-url', default='env://', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--seed', type=int, default=42,
                    help="random seed for initialization")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
args = parser.parse_args()

if len(args.saving_folder) == 0:
    args.saving_folder = None

if "WORLD_SIZE" in os.environ:
    args.world_size = int(os.environ["WORLD_SIZE"])
args.distributed = args.world_size > 1
args.gpu = 0
args.n_gpu = torch.cuda.device_count()

# Setup CUDA, GPU & distributed training
if args.distributed:
    if args.local_rank != -1: # for torch.distributed.launch
        args.rank = args.local_rank
        args.gpu = args.local_rank
    elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank, timeout=timedelta(minutes=60))

if args.distributed:
    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        device = torch.device("cuda", args.gpu)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device

# if args.local_rank != -1:
#     from apex.parallel import DistributedDataParallel as DDP

# suppress printing if not on master gpu
if args.rank not in  [-1, 0]:
    def print_pass(*args):
        pass
    builtins.print = print_pass

if args.saving_folder != None and not os.path.isdir(args.saving_folder):
    os.makedirs(args.saving_folder, exist_ok=True)

if args.log_saving_directory != None:
    os.makedirs(os.path.join(args.saving_folder, args.log_saving_directory), exist_ok=True)

# Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
                    filename=os.path.join(args.saving_folder, "logs", args.name))

logger = logging.getLogger(__name__)
logger.warning("Process rank: %s, rank: %s, gpu: %s, device: %s, n_gpu: %s, distributed training: %s" %
               (args.local_rank, args.rank, args.gpu, args.device, args.n_gpu, bool(args.distributed)))

#set the seed
set_seed(args)

if args.rank in [-1, 0]:
    logger.info("***** Loading Data *****")
#start preparing the dataset
# transform_train = transforms.Compose([
#         transforms.RandomResizedCrop((224, 224), scale=(0.05, 1.0)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
# ])
# transform_test = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
# ])
#
# trainset = datasets.CIFAR10(root="./data/datasets",
#                                 train=True,
#                                 download=False,
#                                 transform=transform_train)
#
# testset = datasets.CIFAR10(root="./data/datsets",
#                             train=False,
#                             download=False,
#                             transform=transform_test)

# Press the green button in the gutter to run the script.
datasets = utils.get_datasets(args.dataset,sliding_window=args.sliding_window, validation=args.validation, window_size=args.window_size,
                              step=args.step_size, downsample=args.downsample_factor, pretraining=args.pretrain)
trainset = datasets['training_set']
validset = datasets['validation_set']
testset = datasets['testing_set']

if args.rank in [-1, 0]:
    logger.info("***** Loading DataLoaders *****")
    logger.info(f"trainset shape: {trainset.data['inputs'].shape}")
    logger.info(f"validation shape: {validset.data['inputs'].shape}")
    logger.info(f"testset shape: {testset.data['inputs'].shape}")
if args.local_rank == 0:
    torch.distributed.barrier()


if args.balance_batch:
  #batch_sampler = BalancedBatchSampler(trainset.data['targets'])
  #train_sampler = batch_sampler if not args.distributed else DistributedSamplerWrapper(batch_sampler,
  #                                                                                     num_replicas=args.world_size,
  #                                                                                     rank=args.rank)

  class_sample_count = np.array(np.unique(trainset.data['targets'], return_counts=True))[1]
  weight = 1. / class_sample_count
  samples_weight = np.array([weight[t] for t in trainset.data['targets']])
  samples_weight = torch.from_numpy(samples_weight)
  batch_sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight), replacement=True)

  train_sampler = batch_sampler if not args.distributed else DistributedSamplerWrapper(batch_sampler, num_replicas=args.world_size, rank=args.rank)
  
else:
	train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)

test_sampler = SequentialSampler(testset)
valid_sampler = SequentialSampler(validset)

training_loader = DataLoader(dataset=trainset,
                             batch_size=args.train_batch_size,
                             sampler=train_sampler,
                             num_workers=args.num_workers,
                             drop_last=True,
                             pin_memory=True)

testing_loader = DataLoader(dataset=testset,
                            batch_size=args.eval_batch_size,
                            sampler=test_sampler,
                            num_workers=args.num_workers,
                            drop_last=True,
                            pin_memory=True)

validation_loader = DataLoader(dataset=validset,
                              batch_size=args.eval_batch_size,
                              sampler=valid_sampler,
                              num_workers=args.num_workers,
                              drop_last=True,
                              pin_memory=True) if args.validation else None

random_sample = next(iter(training_loader))
if args.rank in [-1, 0]:
    logger.info("input shape: %s, label shape: %s" %
                   (random_sample[0].shape, random_sample[1].shape))
    logger.info("train_loader length: %s" % (len(training_loader)))

#setup the model
model = None
num_classes = datasets['training_set'].num_classes
if args.rank in [-1, 0]:
    logger.info("Number of classes: %s" % (num_classes))
    logger.info("Patch size: %s" % (args.mlp_patch_size))
    logger.info("Token dim: %s" % (args.mlp_token_dim))
    logger.info("Channel dim: %s" % (args.mlp_channel_dim))
    logger.info("Patch dim: %s" % (args.mlp_patch_dim))
    logger.info("Num blocks/layers: %s" % (args.mlp_num_blocks))
    logger.info("Sliding window: %s" % (args.window_size))
    logger.info("Window step size: %s" % (args.step_size))
    logger.info("Balanced: %s" % (args.balance_batch))

if args.pretrain:
    config = CONFIGS[args.model_type]
    model = MLPMixer.MlpMixer(image_height=224, image_width=224,
                              patch_size=config.patch_size, token_dim=config.token_mlp_dim,
                              channel_dim=config.channel_mlp_dim,
                              patch_dim=config.patch_dim, num_classes=num_classes,
                              num_blocks=config.num_blocks)
    model.load_pretrained(np.load(args.pretrained_file))
else:
    model = MLPMixer.MlpMixer(image_height=random_sample[0].shape[1], image_width=random_sample[0].shape[2],
                              patch_size=args.mlp_patch_size, token_dim=args.mlp_token_dim,
                              channel_dim=args.mlp_channel_dim,
                              patch_dim=args.mlp_patch_dim, num_classes=num_classes,
                              num_blocks=args.mlp_num_blocks, NoToken=args.mlp_no_token,
                              NoChannel=args.mlp_no_channel, NoRGB=args.mlp_no_RGB_embed)

#set the model to its represpective device
model.to(args.device)
if args.distributed:
    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    if args.gpu is not None:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    else:
        model = torch.nn.parallel.DistributedDataParallel(model)
        model_without_ddp = model.module

#initialize the trainer
trainer = train.Trainer(model=model, training_loader=training_loader, validation_loader=validation_loader,
                        test_loader=testing_loader, verbose=args.verbose, saving_folder=args.saving_folder,
                        num_classes=num_classes, id2label=testset.id2label)
#train the model
trainer.train(args)

#test the model
if args.rank in [-1, 0]:
    #current model
    trainer.validate(args, test=True)

    #best model
    trainer.load_model(args.name)
    trainer.validate(args, test=True, best=True)

# if args.saving_folder != None:
#   with open(os.path.join(args.saving_folder,'run_info.json'), 'w') as write_file:
#       json.dump(vars(args), write_file)
