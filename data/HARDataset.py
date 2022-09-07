import pandas as pd
import numpy as np
import seaborn as sb
import torch
from scipy import stats

class HAR_dataset():
    def __init__(self, path, transform, target_transform):
        self.path = path
        self.transform = transform
        self.target_transform = target_transform
        self.num_classes = 1

    def generate_data(self, window_size=30, step_size=10, downsample=1, downsample_ratio=1.0, slide_window=True):
        self.window_size = window_size
        self.read_data(downsample=downsample)
        
        if slide_window:
            inputs = []
            targets = []
            counter = 0
            while counter + window_size < len(self.data['inputs']):
                inputs += [self.data['inputs'][counter:window_size + counter, :]]
                targets += [stats.mode(self.data['targets'][counter:window_size + counter], axis=None).mode[0]]
                counter += step_size
            self.data = {'inputs': np.asarray(inputs).transpose(0, 2, 1), 'targets': np.asarray(targets, dtype=int)}

        # if slide_window:
        #     inputs = self.sliding_window(array=self.data['inputs'], window_size=window_size, step_size=step_size, downsampling_ratio=downsample_ratio)
        #     targets = self.sliding_window(array=self.data['targets'], window_size=window_size, step_size=step_size, downsampling_ratio=downsample_ratio)
        #
        #     # get the modal (most common) target from each row (use the lowest value there are more than one common value)
        #     targets = stats.mode(targets, axis=1).mode[:, 0]
        #
        #     #use the final element in each window as the target
        #     #targets = targets[:, -1]
        #
        #     #let each row in the sample represent the time series of a single feature
        #     self.data['inputs'] = np.asarray(inputs).transpose(0, 2, 1)
        #     self.data['targets'] = np.asarray(targets, dtype=int)
    def sliding_window(self, array, window_size, step_size, downsampling_ratio=1, start_index=0):
        '''
            add sliding window functionality over the data in any number of dimensions
            Parameters:
                window_size - an int (a is 1D) or tuple (a is 2D or greater) representing the size
                     of each dimension of the window
                step_size - an int (a is 1D) or tuple (a is 2D or greater) representing the
                     amount to slide the window in each dimension. If not specified, it
                     defaults to ws.
                clearing_time_index - and int representing the
                    index to start the sliding window from
                downsampling_ratio - this applies the downsamplign ratio effect
            '''
        if downsampling_ratio >= 1.0:
            start = start_index
            max_time_steps = (((array.shape[0] - window_size) // step_size))

            K_indices = np.arange(0, window_size * downsampling_ratio, step=downsampling_ratio)
            T_indices = np.arange(0, (max_time_steps + 1) * downsampling_ratio, step=downsampling_ratio)

            sub_windows = np.round(
                start +
                np.expand_dims(K_indices, 0) +
                np.expand_dims(T_indices, 0).T
            ).astype(np.int)

            return array[sub_windows]
        else:
            start = start_index
            max_time_steps = (((array.shape[0] - window_size) // step_size))

            sub_windows = (
                    start +
                    np.expand_dims(np.arange(window_size), 0) +
                    # Create a rightmost vector as [0, V, 2V, ...].
                    np.expand_dims(np.arange(max_time_steps + 1, step=step_size), 0).T
            )

            return array[sub_windows]

    def __getitem__(self, index):
        input, target = self.data['inputs'][index], self.data['targets'][index].reshape(1, -1)

        if self.transform is not None:
            input = self.transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return input, target

    def input_shape(self):
        return self.data['inputs'].shape[1::]

    def __len__(self):
        return len(self.data['inputs'])

    def get_data_weights(self):
        class_count = np.zeros((self.num_classes,)).astype(float)
        for i in range(self.num_classes):
            class_count[i] = len(np.where(self.data['targets'] == i)[0])
        weights = (1 / torch.from_numpy(class_count).type(torch.DoubleTensor))
        return weights
