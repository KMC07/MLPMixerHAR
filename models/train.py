import _pickle as pickle
import itertools
import logging
import os
import random
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from .scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from .dist_utils import get_world_size

logger = logging.getLogger(__name__)


def set_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.n_gpu > 0:
		torch.cuda.manual_seed_all(args.seed)


class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


class Trainer():
	def __init__(self, model, training_loader, validation_loader=None, test_loader=None, verbose=False,
	             saving_folder=None, num_classes=5, id2label=None):
		self.model = model
		self.training_loader = training_loader
		self.validation_loader = validation_loader
		self.test_loader = test_loader
		self.verbose = verbose
		self.saving_folder = saving_folder
		self.num_classes = num_classes
		self.id2label = id2label

		self.writer = None
		self.losses = None
		self.optimizer = None
		self.epoch_iterator = None
		self.criterion = None
		self.scheduler = None

		self.global_step = 0
		self.best_acc = 0

	def validate(self, args, test=True, cm=False, best=False):
		# Validation!
		process = ""
		current_process = ""
		if not test:
			if not self.validation_loader:
				return
			test_loader = self.validation_loader
			process = "validation"
			current_process = "Validating"
		else:
			test_loader = self.test_loader
			process = "Test"
			current_process = "Testing"

		# create the evaluation loss class
		eval_losses = AverageMeter()

		# log the validation process
		logger.info(f"***** Running {process} *****")
		logger.info("  Num steps = %d", len(test_loader))
		logger.info("  Batch size = %d", args.eval_batch_size)

		# start evaluating the model on the repsective dataset loader
		self.model.eval()
		all_preds, all_label = [], []
		if self.verbose:
			epoch_iterator = tqdm(test_loader,
			                      desc=f"{current_process}... (loss=X.X)",
			                      bar_format="{l_bar}{r_bar}",
			                      dynamic_ncols=True,
			                      disable=args.local_rank not in [-1, 0])
		else:
			epoch_iterator = test_loader

		validate_loss = nn.CrossEntropyLoss()
		total_time = 0
		repetitions = 0
		for step, batch in enumerate(epoch_iterator):
			repetitions += 1
			total_time += self.validate_batch(args.device, batch, validate_loss, eval_losses, all_preds, all_label)
			if self.verbose:
				epoch_iterator.set_description(f"{current_process}... (loss=%2.5f)" % eval_losses.val)

		all_preds, all_label = all_preds[0], all_label[0]
		accuracy = self.simple_accuracy(torch.from_numpy(all_preds), torch.from_numpy(all_label).flatten())
		microf1 = f1_score(all_label, all_preds, average='micro')
		macrof1 = f1_score(all_label, all_preds, average='macro')
		weightedf1 = f1_score(all_label, all_preds, average='weighted')
		throughput = (repetitions * args.eval_batch_size)/total_time

		logger.info("\n")
		logger.info(f"{process} Results")
		logger.info("Global Steps: %d" % self.global_step)
		logger.info(f"{process} Loss: %2.5f" % eval_losses.avg)
		logger.info(f"{process} Accuracy: %2.5f" % accuracy)
		logger.info(f"{process} Micro f1: %2.5f" % microf1)
		logger.info(f"{process} Macro f1: %2.5f" % macrof1)
		logger.info(f"{process} Weighted f1: %2.5f" % weightedf1)
		logger.info(f"{process} Throughput: %2.5f" % throughput)

		# store the accuaracy
		if args.rank in [-1, 0] and self.writer:
			self.writer.add_scalar(f"{process}/accuracy", scalar_value=accuracy, global_step=self.global_step)
			self.writer.add_scalar(f"{process}/micro f1", scalar_value=microf1, global_step=self.global_step)
			self.writer.add_scalar(f"{process}/macro f1", scalar_value=macrof1, global_step=self.global_step)
			self.writer.add_scalar(f"{process}/weighted f1", scalar_value=weightedf1, global_step=self.global_step)

		if test:
			self.writer.add_scalar(f"{process}/throughput", scalar_value=throughput, global_step=self.global_step)

		if cm:
			cm = confusion_matrix(all_label, all_preds)
			self.plot_confusion_matrix(cm, classes=self.id2label, normalize=True)

		if args.rank in [-1, 0] and best:
			self.writer.add_hparams({
				'patch_size': args.mlp_patch_size,
				'token_dim': args.mlp_token_dim,
				'channel_dim': args.mlp_channel_dim,
				'patch_dim': args.mlp_patch_dim,
				'num_blocks': args.mlp_num_blocks,
				'sliding_window': args.window_size,
				'window_step_size': args.step_size,
				'balanced': args.balance_batch,
			},{
				'micro f1': microf1,
				'macro f1': macrof1,
				'weightedf1': weightedf1,
			})

		best_metric = {"accuracy": accuracy, "macro_f1": macrof1, "micro_f1": microf1, "weighted_f1": weightedf1}
		return best_metric[args.save_best_metric]

	def validate_batch(self, device, batch, validate_loss, eval_losses, all_preds, all_label):
		batch = tuple(t.to(device) for t in batch)
		x, y = batch
		with torch.no_grad():
			starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
			starter.record()
			logits = self.model(x)
			ender.record()
			torch.cuda.synchronize()
			curr_time = starter.elapsed_time(ender) / 1000

			eval_loss = validate_loss(logits, y.flatten())
			eval_losses.update(eval_loss.item())

			preds = torch.argmax(logits, dim=-1)

		if len(all_preds) == 0:
			all_preds.append(preds.cpu().numpy().reshape(-1))
			all_label.append(y.data.cpu().numpy().reshape(-1))
		else:
			all_preds[0] = np.append(
				all_preds[0], preds.cpu().numpy().reshape(-1)
			)
			all_label[0] = np.append(
				all_label[0], y.data.cpu().numpy().reshape(-1), axis=0
			)

		return curr_time

	def train(self, args):
		""" Train the model """
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, betas=(0.9, 0.999),
		                                  weight_decay=args.weight_decay, eps=1e-08)
		# self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum,
		#                                  weight_decay=args.weight_decay)
		self.criterion = nn.CrossEntropyLoss()

		if args.rank in [-1, 0]:
			os.makedirs(args.saving_folder, exist_ok=True)
			if args.summary_saving_directory != None:
				os.makedirs(os.path.join(args.saving_folder, args.summary_saving_directory), exist_ok=True)
			pretrain_str = "_pretrained" if args.pretrain else ""
			balanced_str = "_balanced" if args.balance_batch else ""
			self.writer = SummaryWriter(
				log_dir=os.path.join(args.saving_folder, args.summary_saving_directory, args.name),
				comment=f"{pretrain_str}{balanced_str}_{self.optimizer.__class__.__name__}")

		 
		args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

		t_total = args.nb_steps
		if args.decay_type == "cosine":
			self.scheduler = WarmupCosineSchedule(self.optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
		else:
			self.scheduler = WarmupLinearSchedule(self.optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

		# Train!
		logger.info("***** Running training *****")
		logger.info("  Total optimization steps = %d", args.nb_steps)
		logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
		logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
		            args.train_batch_size * args.gradient_accumulation_steps * (
			            torch.distributed.get_world_size() if args.local_rank != -1 else 1))
		logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

		self.model.zero_grad()
		set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
		self.losses = AverageMeter()
		self.global_step, self.best_metric = 0, 0

		while True:
			if self.train_step(args, t_total):
				break

		if args.rank in [-1, 0]:
			self.writer.close()
		logger.info(f"Best {args.save_best_metric}: \t%f" % self.best_metric)
		logger.info("End Training!")

		if args.rank in [-1, 0] and args.save_final_model:
			self.save_model(f"{args.name}_final")

	def train_step(self, args, t_total):
		self.model.train()
		if self.verbose:
			self.epoch_iterator = tqdm(self.training_loader,
			                           desc="Training (X / X Steps) (loss=X.X)",
			                           bar_format="{l_bar}{r_bar}",
			                           dynamic_ncols=True,
			                           disable=args.local_rank not in [-1, 0])
		else:
			self.epoch_iterator = self.training_loader

		for step, batch in enumerate(self.epoch_iterator):
			if self.train_batch(args, step, batch, t_total):
				break

		self.losses.reset()
		if self.global_step % t_total == 0:
			return True
		return False

	def train_batch(self, args, step, batch, t_total):
		batch = tuple(t.to(args.device) for t in batch)
		x, y = batch
		loss = self.model(x)

		# calulate the training accuracy
		if args.rank in [-1, 0]:
			preds = torch.argmax(loss, dim=-1)
			training_acc = self.simple_accuracy(preds, y)
			self.writer.add_scalar("train/acc", scalar_value=training_acc, global_step=self.global_step)

		if y is not None:
			loss = self.criterion(loss.view(-1, self.num_classes), y.view(-1))

		if args.gradient_accumulation_steps > 1:
			loss = loss / args.gradient_accumulation_steps
		loss.backward()

		if (step + 1) % args.gradient_accumulation_steps == 0:
			self.losses.update(loss.item() * args.gradient_accumulation_steps)
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
			self.optimizer.step()
			self.optimizer.zero_grad()
			self.scheduler.step()
			self.global_step += 1

			if self.verbose:
				self.epoch_iterator.set_description(
					"Training (%d / %d Steps) (loss=%2.5f)" % (self.global_step, t_total, self.losses.val)
				)

			if args.rank in [-1, 0]:
				self.writer.add_scalar("train/loss", scalar_value=self.losses.val, global_step=self.global_step)
				self.writer.add_scalar("train/lr", scalar_value=self.scheduler.get_last_lr()[0],
				                       global_step=self.global_step)
			if self.global_step % args.eval_every == 0 and args.rank in [-1, 0]:
				metric = self.validate(args, test=False)
				if self.best_metric < metric:
					self.save_model(args.name)
					self.best_metric = metric
				self.model.train()

			if self.global_step % t_total == 0:
				return True
		return False

	def simple_accuracy(self, preds, labels):
		return torch.mean(torch.eq(preds, labels).float(), dtype=torch.float)

	def save_model(self, name=''):
		model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
		model_checkpoint = os.path.join(self.saving_folder, "%s_checkpoint.bin" % name)
		torch.save(model_to_save.state_dict(), model_checkpoint)
		logger.info("Saved model checkpoint to [DIR: %s]", self.saving_folder)

	def load_model(self, name=''):
		model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
		model_checkpoint = os.path.join(self.saving_folder, "%s_checkpoint.bin" % name)
		model_to_load.load_state_dict(torch.load(model_checkpoint))
		logger.info("Loaded model checkpoint from [DIR: %s]", self.saving_folder)

	def plot_confusion_matrix(self, cm, classes,
	                          normalize=False,
	                          title='Confusion matrix',
	                          cmap=plt.cm.Blues):
		if normalize:
			cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
			# print("Normalized confusion matrix")
		else:
			# print('Confusion matrix, without normalization')
			pass

		# print(cm)

		# Calculate chart area size
		leftmargin = 0.5  # inches
		rightmargin = 0.5  # inches
		categorysize = 0.5  # inches
		figwidth = leftmargin + rightmargin + (len(classes) * categorysize)

		f = plt.figure(figsize=(figwidth, figwidth))

		# Create an axes instance and ajust the subplot size
		ax = f.add_subplot(111)
		ax.set_aspect(1)
		f.subplots_adjust(left=leftmargin / figwidth, right=1 - rightmargin / figwidth, top=0.94, bottom=0.1)

		res = ax.imshow(cm, interpolation='nearest', cmap=cmap)

		plt.imshow(cm, interpolation='nearest', cmap=cmap)
		plt.title(title)
		plt.colorbar()
		ax.set_xticks(range(len(classes)))
		ax.set_yticks(range(len(classes)))
		ax.set_xticklabels(classes, rotation=90, ha='right')
		ax.set_yticklabels(classes)

		fmt = '.2f' if normalize else 'd'
		thresh = cm.max() / 2.
		for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
			plt.text(j, i, format(cm[i, j], fmt),
			         horizontalalignment="center",
			         color="white" if cm[i, j] > thresh else "black")
		plt.tight_layout()
		np.set_printoptions(precision=3)

		plt.xlabel('Predicted')
		plt.ylabel('Actual')

		plt.savefig(os.path.join(self.saving_folder, 'confusion_matrix.png'), bbox_inches='tight')

	def print_last_step(self):
		pass
