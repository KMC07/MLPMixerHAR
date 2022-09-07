import numpy as np
import torch

from operator import itemgetter
from torch.utils.data import Dataset, DistributedSampler, Sampler
from torch.utils.data.sampler import BatchSampler, WeightedRandomSampler
from typing import Iterator, List, Optional, Union

class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)

class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
                distributed training
            rank (int, optional): Rank of the current process
                within ``num_replicas``
            shuffle (bool, optional): If true (default),
                sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.
        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


class BalancedBatchSampler(Sampler):
    def __init__(self, labels):
        self.labels = labels
        self.nb_classes = int(max(labels)+1)
        self.build_classes_iterators()
    def __iter__(self):
        return iter(self.merged_iterator())
    def __len__(self):
        return len(self.labels)
    def build_classes_iterators(self):
        iterators = []
        classes_indexes = []
        for i in range(self.nb_classes):
            classes_indexes += [np.where(self.labels == i)[0]]
            permutation = np.random.permutation(len(classes_indexes[-1]))
            iterators += [iter(classes_indexes[-1][permutation])]
        self.classes_indexes = classes_indexes
        self.classes_iterators = iterators
    def merged_iterator(self):
        counter = 0
        while counter < len(self.labels):
            next_index = next(self.classes_iterators[0],None)
            if next_index != None:
                yield next_index
                counter += 1
            else:
                self.buld_class_iterator(0)
                next_index = next(self.classes_iterators[0])
                yield next_index
                counter += 1
            for j,iterator in enumerate(self.classes_iterators):
                next_index = next(iterator,None)
                if next_index != None:
                    yield next_index
                    counter += 1
                else:
                    self.buld_class_iterator(j)
                    next_index = next(self.classes_iterators[j])
                    yield next_index
                    counter += 1
    def buld_class_iterator(self,label):
        permutation = np.random.permutation(len(self.classes_indexes[label]))
        self.classes_iterators[label] = iter(self.classes_indexes[label][permutation])