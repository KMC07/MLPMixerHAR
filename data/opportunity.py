import numpy as np
import pandas as pd
import os
import csv
import torch
from pathlib import Path
from .HARDataset import HAR_dataset

default_path = './datasets/OpportunityUCIDataset/'
default_frequency = 30
default_test_indexes = ['S2-ADL4.dat','S2-ADL5.dat','S3-ADL4.dat','S3-ADL5.dat']

class OpportunityDataset(HAR_dataset):
    def __init__(self, fileIndex, path=default_path, transform=None, target_transform=None, pretraining=False, locomotion=False, drop=False):
        super(OpportunityDataset, self).__init__(path=path, transform=transform, target_transform=target_transform)
        self.pretraining = pretraining
        self.locomotion = locomotion
        self.drop = drop
        self.files = np.asarray([['S1-ADL1.dat','S1-ADL2.dat','S1-ADL3.dat', 'S1-ADL4.dat', 'S1-ADL5.dat', 'S1-Drill.dat'],
                      ['S2-ADL1.dat', 'S2-ADL2.dat','S2-ADL3.dat', 'S2-ADL4.dat','S2-ADL5.dat', 'S2-Drill.dat'],
                      ['S3-ADL1.dat', 'S3-ADL2.dat','S3-ADL3.dat', 'S3-ADL4.dat','S3-ADL5.dat', 'S3-Drill.dat'],
                      ['S4-ADL1.dat', 'S4-ADL2.dat', 'S4-ADL3.dat', 'S4-ADL4.dat', 'S4-ADL5.dat', 'S4-Drill.dat']])
        self.files = [self.files[i,j] for i,j in fileIndex]

        # Hardcoded thresholds to define global maximums and minimums for every one of the 113 sensor channels employed in the
        # OPPORTUNITY challenge
        self.NORM_MAX_THRESHOLDS = [3531., 4270., 4535.,  1524.,  2948.,  2306.,  3705.,  2005.,
                                    1933., 2482., 3511.,  4459.,  2762.,  2867.,  2728.,  1700.,
                                    2830., 1624., 3479.,  152195.,40904., 4371.,  4172.,  4772.,
                                    1585., 2437., 2516.,  2691.,  3460.,  3400.,  2075.,  2372.,
                                    2966., 3329., 4710.,  4533.,  507.,   1170.,  1307.,  3737.,
                                    3398., 1903., 1591.,  1112.,  1202.,  665.,   1812.,  1847.,
                                    8289., 7011., 4733.,  1763.,  1378.,  1315.,  1887.,  1821.,
                                    1911., 10482.,7298.,  7509.,  1748.,  1317.,  3368.,  895.,
                                    4473., 4087., 9767.,  6121.,  5799.,  1607.,  1337.,  922.,
                                    3387., 1390., 4010.,  9454.,  6434.,  7956.,  2166.,  1616.,
                                    1688., 280.,  64.,    322.,   7175.,  6050.,  7347.,  6875.,
                                    6516., 6769., 12946., 14765., 11713., 14765., 33444., 11713.,
                                    318.,  270.,  59.,    288.,   8390.,  10767., 9902.,  8411.,
                                    7356., 6752., 12467., 16091., 10720., 16091., 17229., 10720.,
                                    272.]

        self.NORM_MIN_THRESHOLDS = [-2750., -1046., -3602., -1602., -472., -2009., -1363., -868.,
                                    -1295., -4092., -4237., -4150., -2601., -2214., -2961., -1170.,
                                    -1566., -1967., -3589., -36024., -9744., -4915., -2236., -2225.,
                                    -1822., -757., -1539., -1888., -4356., -1101., -2334., -1806.,
                                    -1454., -4958., -6308., -5271., -1930., -1140., -903., -3344.,
                                    -2978., -1888., -823., -938., -1246., -1898., -1917., -1662.,
                                    -7057., -4353., -4725., -930., -1538., -1256., -1868., -1808.,
                                    -1850., -11966., -9206., -8462., -1514., -2846., -1622., -3038.,
                                    -4264., -1360., -13442., -5571., -4161., -1288., -967., -1383.,
                                    -4443., -5007., -3400., -10883., -6078., -6405., -2637., -2001.,
                                    -1454., -282., -92., -296., -8895., -11325., -7630., -7020.,
                                    -7094., -6736., -33444., -16946., -9610., -16946., -12946., -9610.,
                                    -287., -277., -91., -292., -6274., -8609., -8174., -5709.,
                                    -9009., -7153., -17229., -13377., -15677., -13377., -12467., -15677.,
                                    -342.]

    def read_data(self,downsample=1, chop_non_related_activities=False):
        files = self.files.copy()
        label_map = []
        if self.locomotion:
            label_map = [
                (0, 'Other'),
                (1, 'Stand'),
                (2, 'Walk'),
                (4, 'Sit'),
                (5, 'Lie'),
            ]
        else:
            label_map = [
                (0, 'Other'),
                (406516, 'Open Door 1'),
                (406517, 'Open Door 2'),
                (404516, 'Close Door 1'),
                (404517, 'Close Door 2'),
                (406520, 'Open Fridge'),
                (404520, 'Close Fridge'),
                (406505, 'Open Dishwasher'),
                (404505, 'Close Dishwasher'),
                (406519, 'Open Drawer 1'),
                (404519, 'Close Drawer 1'),
                (406511, 'Open Drawer 2'),
                (404511, 'Close Drawer 2'),
                (406508, 'Open Drawer 3'),
                (404508, 'Close Drawer 3'),
                (408512, 'Clean Table'),
                (407521, 'Drink from Cup'),
                (405506, 'Toggle Switch')
            ]

        ##convert the label map into ordered labelled id
        label2id = {x[0] : i for i, x in enumerate(label_map)}

        # select the columns that are being used in the opportunity dataset
        # cols = [i for i in range(1, 251)]

        # these are the columns to remove from the opportunity dataset
        # remove_features = np.arange(1)
        # remove_features = np.concatenate([remove_features, np.arange(46, 50)])
        # remove_features = np.concatenate([remove_features, np.arange(59, 63)])
        # remove_features = np.concatenate([remove_features, np.arange(72, 76)])
        # remove_features = np.concatenate([remove_features, np.arange(85, 89)])
        # remove_features = np.concatenate([remove_features, np.arange(98, 102)])
        # remove_features = np.concatenate([remove_features, np.arange(134, 243)])
        # remove_features = np.concatenate([remove_features, np.arange(244, 249)])
        #
        # # drop the columns from the dataframe
        # cols = np.delete(cols, remove_features)

        cols = [
            38, 39, 40, 41, 42, 43, 44, 45, 46, 51, 52, 53, 54, 55, 56, 57, 58, 59,
            64, 65, 66, 67, 68, 69, 70, 71, 72, 77, 78, 79, 80, 81, 82, 83, 84, 85,
            90, 91, 92, 93, 94, 95, 96, 97, 98, 103, 104, 105, 106, 107, 108, 109,
            110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124,
            125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 244, 250]
        cols = [x - 1 for x in cols]  # labels for 18 activities (including other)

        #read the files
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
            self.data['inputs'] = self.data['inputs'][::downsample,:]
            self.data['targets'] = self.data['targets'][::downsample]

    def normalize_data(self, data):
        """Normalizes all sensor channels
            :param data: numpy integer matrix
                Sensor data
            :return:
                Normalized sensor data
            """
        diffs = np.array(self.NORM_MAX_THRESHOLDS) - np.array(self.NORM_MIN_THRESHOLDS)
        for i in np.arange(data.shape[1]):
            data[:, i] = (data[:, i] - self.NORM_MIN_THRESHOLDS[i]) / diffs[i]
        #     Checking the boundaries
        data[data > 1] = 0.99
        data[data < 0] = 0.00
        return data

    def read_files(self, filelist, cols, label2id):
        data = np.empty((0, 77))
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

            data = np.vstack((data, df.iloc[:, :-2]))
            #data = np.vstack((data, df.iloc[:, :-2]/1000))
            ##determine whether to use locomotion labels as targets or the gestures as targets
            if self.locomotion:
                labels = np.concatenate((labels, df.iloc[:, -2].map(label2id)), axis=None)
            else:
                labels = np.concatenate((labels, df.iloc[:, -1].map(label2id)), axis=None)

        data = self.normalize_data(data)

        if self.pretraining:
            data = np.pad(array=data, pad_width=([(0, 0), (0, 147)]), mode='constant', constant_values=0)
        self.data = {'inputs': np.asarray(data), 'targets': np.asarray(labels, dtype=int)}