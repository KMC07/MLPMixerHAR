import numpy as np
import torch
import torchvision.transforms as transforms

def get_dataset_info(dataset):
	if dataset == 'opportunity_locomotion':
	    from data import opportunity
	    dataset = opportunity.OpportunityDataset
	    val_id = [[0,1]]
	    test_id = [[1,3],[1,4],[2,3],[2,4]]
	    train_id = [[0,0],[0,2],[0,3],[0,4],[0,5],
	                [1,0],[1,1],[1,2],[1,5],
	                [2,0],[2,1],[2,2],[2,5],
	                [3,0],[3,1],[3,2],[3,3],[3,4],[3,5]]
	    kwargs = {'locomotion': True}

	elif dataset == 'opportunity_gestures':
	    from data import opportunity
	    dataset = opportunity.OpportunityDataset
	    val_id = [[0,1]]
	    test_id = [[1,3],[1,4],[2,3],[2,4]]
	    train_id = [[0,0],[0,2],[0,3],[0,4],[0,5],
	                [1,0],[1,1],[1,2],[1,5],
	                [2,0],[2,1],[2,2],[2,5],
	                [3,0],[3,1],[3,2],[3,3],[3,4],[3,5]]
	    kwargs = {'locomotion': False}

	elif dataset == 'pamap2':
		from data import pamap2
		dataset = pamap2.PAMAP2Dataset
		val_id = [4]
		test_id = [5]
		train_id = [0, 1, 2, 3, 6, 7]
		kwargs = {}

	elif dataset == 'daphnet':
		from data import daphnet
		dataset = daphnet.DaphNetDataset
		val_id = [8]
		test_id = [1]
		train_id = [0, 2, 3, 4, 5, 6, 7, 9]
		kwargs = {}

	return {'dataset' : dataset, 'train' : train_id, 'test' : test_id, 'validation' : val_id, "kwargs": kwargs}


def data_transforms_float(x):
	return torch.from_numpy(x).type(torch.FloatTensor)

def data_transforms_long(x):
	return torch.from_numpy(x).type(torch.LongTensor)

def get_datasets(dataset, train=True, validation=True, Test=True, sliding_window=True, window_size=90, step=45, downsample=1, pretraining=False):
	dataset_info = get_dataset_info(dataset)
	input_transform = transforms.Compose([
		transforms.Lambda(data_transforms_float)
	])
	target_transform = transforms.Compose([
		transforms.Lambda(data_transforms_long)
	])

	test = dataset_info['dataset'](fileIndex=dataset_info['test'], transform=input_transform, target_transform=target_transform, pretraining=pretraining, **dataset_info['kwargs'])
	test.generate_data(window_size=window_size, step_size=step, downsample=downsample, slide_window=sliding_window)
	y = {'testing_set': test, 'training_set': None, 'validation_set': None}

	if train:
		trainfiles = dataset_info['train'] if validation else dataset_info['train'] + dataset_info['validation']
		training = dataset_info['dataset'](fileIndex=trainfiles, transform=input_transform, target_transform=target_transform, pretraining=pretraining, **dataset_info['kwargs'])
		training.generate_data(window_size=window_size, step_size=step, downsample=downsample, slide_window=sliding_window)
		y['training_set'] = training

	if validation:
		val = dataset_info['dataset'](fileIndex=dataset_info['validation'], transform=input_transform, target_transform=target_transform, pretraining=pretraining, **dataset_info['kwargs'])
		val.generate_data(window_size=window_size, step_size=step, downsample=downsample, slide_window=sliding_window)
		y['validation_set'] = val
	return y