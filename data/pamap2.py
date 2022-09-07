import numpy as np
import pandas as pd
import os
import csv
import torch
from pathlib import Path
from .HARDataset import HAR_dataset

default_path = './datasets/PAMAP2Dataset/'
default_frequency = 100
default_test_index = 5
heart_rate_index = [2]
acc16_indexes = [4,5,6,21,22,23,38,39,40]

class PAMAP2Dataset(HAR_dataset):
    def __init__(self, fileIndex, path=default_path, transform=None, target_transform=None, pretraining=False):
        super(PAMAP2Dataset,self).__init__(path=path,transform=transform,target_transform=target_transform)
        self.pretraining = pretraining
        self.files = ['subject101.dat', 
                      'subject102.dat',
                      'subject103.dat',
                      'subject104.dat',
                      'subject105.dat',
                      'subject106.dat',
                      'subject107.dat',
                      'subject108.dat']
        self.files = [self.files[i] for i in fileIndex]

        colNames = ["timestamp", "activityID", "heartrate"]

        IMUhand = ['handTemperature',
                   'handAcc16_1', 'handAcc16_2', 'handAcc16_3',
                   'handAcc6_1', 'handAcc6_2', 'handAcc6_3',
                   'handGyro1', 'handGyro2', 'handGyro3',
                   'handMagne1', 'handMagne2', 'handMagne3',
                   'handOrientation1', 'handOrientation2', 'handOrientation3', 'handOrientation4']

        IMUchest = ['chestTemperature',
                    'chestAcc16_1', 'chestAcc16_2', 'chestAcc16_3',
                    'chestAcc6_1', 'chestAcc6_2', 'chestAcc6_3',
                    'chestGyro1', 'chestGyro2', 'chestGyro3',
                    'chestMagne1', 'chestMagne2', 'chestMagne3',
                    'chestOrientation1', 'chestOrientation2', 'chestOrientation3', 'chestOrientation4']

        IMUankle = ['ankleTemperature',
                    'ankleAcc16_1', 'ankleAcc16_2', 'ankleAcc16_3',
                    'ankleAcc6_1', 'ankleAcc6_2', 'ankleAcc6_3',
                    'ankleGyro1', 'ankleGyro2', 'ankleGyro3',
                    'ankleMagne1', 'ankleMagne2', 'ankleMagne3',
                    'ankleOrientation1', 'ankleOrientation2', 'ankleOrientation3', 'ankleOrientation4']

        self.columns = colNames + IMUhand + IMUchest + IMUankle  # all columns in one list

    def read_data(self,downsample=1, chop_non_related_activities=True, trim_activities=True):
        files = self.files.copy()
        label_map = [
            (0, 'transient'),
            (1, 'lying'),
            (2, 'sitting'),
            (3, 'standing'),
            (4, 'walking'),
            (5, 'running'),
            (6, 'cycling'),
            (7, 'Nordic walking'),
            # (9, 'watching TV'),
            # (10, 'computer work'),
            # (11, 'car driving'),
            (12, 'ascending stairs'),
            (13, 'descending stairs'),
            (16, 'vacuum cleaning'),
            (17, 'ironing'),
            # (18, 'folding laundry'),
            # (19, 'house cleaning'),
            # (20, 'playing soccer'),
            (24, 'rope jumping')
        ]
        
        label2id = {x[0]: i for i, x in enumerate(label_map)}
        cols = [1] + heart_rate_index + acc16_indexes

        self.read_files(files, cols, label2id)
        self.data['targets'] = np.asarray([int(label2id[i]) for i in self.data['targets'].tolist()]).astype(int)

        #remove all transient related activities as these are relevant to the problem
        if chop_non_related_activities:
            tmp = {'inputs' : [],'targets': []}
            for x,y in zip(self.data['inputs'],self.data['targets']):
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

        if trim_activities:
            self.trim_activities()

        if downsample > 1:
            self.data['inputs'] = self.data['inputs'][::downsample,:]
            self.data['targets'] = self.data['targets'][::downsample]

    def read_files(self, filelist, cols, label2id, interpolate_heart_rate=True):
        data = []
        labels = []
        for i, filename in enumerate(filelist):
            path = default_path + 'Protocol/%s' % filename
            dirname = os.path.dirname(__file__)
            filename = os.path.join(dirname, path)

            current_data = pd.read_csv(filename, delimiter=' ')
            current_data.columns = self.columns

            if interpolate_heart_rate:
                current_data.iloc[:,heart_rate_index] = current_data.iloc[:,heart_rate_index].interpolate()

            current_data = current_data.dropna()
            #current_data = current_data.iloc[:,cols]
            
            current_data = current_data.drop(
            ['timestamp', 'handOrientation1', 'handOrientation2', 'handOrientation3', 'handOrientation4',
             'chestOrientation1', 'chestOrientation2', 'chestOrientation3', 'chestOrientation4',
             'ankleOrientation1', 'ankleOrientation2', 'ankleOrientation3', 'ankleOrientation4'],
            axis=1) # removal of orientation columns as they are not needed
            #current_data = current_data.drop(
            #['timestamp'],
            #axis=1)  # removal of timestamp columns as they are not needed

            data.append(current_data.iloc[:,1::])
            labels.append(current_data.iloc[:,0])

        self.data = {'inputs': np.concatenate([np.asarray(i) for i in data]).astype(float), 'targets': np.concatenate([np.asarray(j) for j in labels]).astype(int)}
        if self.pretraining:
            self.data['inputs'] = np.pad(array=self.data['inputs'], pad_width=([(0, 0), (0, 184)]), mode='constant', constant_values=0)

    def plot_data(self,saving_folder):
        if not os.path.isdir(saving_folder):
            os.mkdir(saving_folder)
        for i,time_series in enumerate(self.data['inputs'].T.tolist()):
            plt.figure()
            plt.plot(time_series)
            plt.savefig(saving_folder+'time_series' + str(i)+'.png')

    def trim_activities(self):
        inputs = []
        targets = []
        trimmed_inputs = []
        trimmed_targets =[]
        for x,y in zip(self.data['inputs'],self.data['targets']):
            if len(targets) == 0:
                targets += [y]
                inputs += [x]
            else:
                if targets[-1] != y:
                    trimmed_inputs += [inputs[1000:-1000]]
                    trimmed_targets += [targets[1000:-1000]]
                    targets = []
                    inputs = []
                else:
                    targets += [y]
                    inputs += [x]
        self.data['inputs'] = np.concatenate(trimmed_inputs)
        self.data['targets'] = np.concatenate(trimmed_targets)