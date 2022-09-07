import numpy as np
import pandas as pd
import os
import csv
import torch
from pathlib import Path
from .HARDataset import HAR_dataset

default_path = './datasets/DaphnetDataset/'
default_frequency = 60
default_test_index = 1
default_validation_index = 8


class DaphNetDataset(HAR_dataset):
    def __init__(self, fileIndex, path=default_path, transform=None, target_transform=None, pretraining=False, drop=True):
        super(DaphNetDataset, self).__init__(path=path, transform=transform, target_transform=target_transform)
        self.drop = drop
        self.pretraining = pretraining
        self.files = np.asarray([['S01R01.txt', 'S01R02.txt'],
                                 ['S02R01.txt', 'S02R02.txt'],
                                 ['S03R01.txt', 'S03R02.txt', 'S03R03.txt'],
                                 ['S04R01.txt'],
                                 ['S05R01.txt', 'S05R02.txt'],
                                 ['S06R01.txt', 'S06R02.txt'],
                                 ['S07R01.txt', 'S07R02.txt'],
                                 ['S08R01.txt'],
                                 ['S09R01.txt'],
                                 ['S10R01.txt']], dtype=object)
        self.files = [self.files[i] for i in fileIndex]

    def read_data(self, downsample=1, chop_non_related_activities=True):
        files = self.files.copy()
        files = [item for sublist in files for item in sublist]

        label_map = [(0,'unrelated'), (1, 'No freeze'), (2, 'freeze')]

        #convert the label map into ordered labelled id
        label2id = {x[0]: i for i, x in enumerate(label_map)}

        cols = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # print "cols",cols
        self.read_files(files, cols, label2id)

        # remove all transient related activities as these are relevant to the problem
        if chop_non_related_activities:
            tmp = {'inputs': [], 'targets': []}
            for x, y in zip(self.data['inputs'], self.data['targets']):
                if int(y) != 0:
                    tmp['inputs'] += [x]
                    tmp['targets'] += [y]
            tmp['inputs'] = np.asarray(tmp['inputs'])
            tmp['targets'] = np.asarray(tmp['targets'])
            tmp['targets'] -= 1
            self.data = tmp
            del label_map[0]

        self.num_classes = len(label_map)

        ##convert the label map into ordered labelled id
        self.label2id = {x[0]: i for i, x in enumerate(label_map)}

        ##convert the label map into an array of the ids
        self.id2label = [x[1] for x in label_map]

        if downsample > 1 and downsample != 0:
            self.data['inputs'] = self.data['inputs'][::downsample, :]
            self.data['targets'] = self.data['targets'][::downsample]

    def read_files(self, filelist, cols, label2id):
        data = np.empty((0, 9))
        labels = np.empty((0))

        for i, filename in enumerate(filelist):
            path = default_path.rstrip('/') + '/dataset/%s' % filename

            dirname = os.path.dirname(__file__)
            filename = os.path.join(dirname, path)

            df = pd.read_csv(filename, header=None, sep=" ")
            df = df[cols]

            if self.drop:
                # drop missing values
                df.dropna(inplace=True)
                if df.shape[0] == 0:
                    continue

            else:
                #peform linear interpolation
                #df.interpolate(method="linear")

                #replace the missing values with zero
                df = df.fillna(0)

            data = np.vstack((data, df.iloc[:, :-1]/1000))
            labels = np.concatenate((labels, df.iloc[:, -1].map(label2id)), axis=None)

        #data = self.normalize_data(data)

        if self.pretraining:
            data = np.pad(array=data, pad_width=([(0, 0), (0, 215)]), mode='constant', constant_values=0)
        self.data = {'inputs': np.asarray(data), 'targets': np.asarray(labels, dtype=int)}

    # def read_files(self, filelist, cols, label2id):
    #     data = []
    #     labels = []
    #     for i, filename in enumerate(filelist):
    #         with open(self.datapath + '/dataset/%s' % filename, 'r') as f:
    #             # print "f",f
    #             reader = csv.reader(f, delimiter=' ')
    #             for line in reader:
    #                 # print "line=",line
    #                 elem = []
    #                 # not including the non related activity
    #                 if line[10] == "0":
    #                     continue
    #                 for ind in cols:
    #                     # print "ind=",ind
    #                     if ind == 10:
    #                         # print "line[ind]",line[ind]
    #                         if line[ind] == "0":
    #                             continue
    #                     elem.append(line[ind])
    #                 if sum([x == 'NaN' for x in elem]) == 0:
    #                     data.append([float(x) / 1000 for x in elem[:-1]])
    #                     labels.append(self.label2id[elem[-1]])
    #
    #     self.data = {'inputs': np.asarray(data), 'targets': np.asarray(labels, dtype=int)}