import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import config as cfg

np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)


def create_dataloader(mode, type=0, snr=0):
    if mode == 'train':
        return DataLoader(
            dataset=Wave_Dataset(mode, type, snr),
            batch_size=cfg.batch,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
            sampler=None
        )
    elif mode == 'valid':
        return DataLoader(
            dataset=Wave_Dataset(mode, type, snr),
            batch_size=cfg.batch, shuffle=False, num_workers=0
        )
    elif mode == 'test':
        return DataLoader(
            dataset=Wave_Dataset(mode, type, snr),
            batch_size=cfg.batch, shuffle=False, num_workers=0
        )


class Wave_Dataset(Dataset):
    def __init__(self, mode, type, snr):
        # load data
        if mode == 'train':
            self.mode = 'train'
            print('<Training dataset>')
            print('Load the data...')
            # self.input_path = './input/to_Noisy_train_dataset.npy'
            # self.input = np.load(self.input_path)
            self.input = []
            for i in range(1, 3):
                self.current_input_path = './input/to_Noisy_train_dataset{}.npy'.format(i)
                self.current_input = np.load(self.current_input_path)
                self.input.extend(self.current_input)
        elif mode == 'valid':
            self.mode = 'valid'
            print('<Validation dataset>')
            print('Load the data...')
            self.input_path = './input/to_Noisy_validation_dataset.npy'
            self.input = np.load(self.input_path)

            # # if you want to use a part of the dataset
            # self.input = self.input[:500]
        elif mode == 'test':
            self.mode = 'test'
            print('<Test dataset>')
            print('Load the data...')
            self.input_path = './input/MS_29m_test_dataset.npy'

            self.input = np.load(self.input_path)
            self.input = self.input[type][snr]

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        if self.mode != 'test' and cfg.perceptual == 'PAM':
            inputs = self.input[idx][0]
            targets = self.input[idx][1][0]
            GMTs = self.input[idx][1][1]

            # transform to torch from numpy
            inputs = torch.from_numpy(inputs)
            targets = torch.from_numpy(targets)
            GMTs = torch.from_numpy(GMTs)

            return inputs, targets, GMTs
        elif cfg.conditional_learning:
            # conditions = self.input[idx][0]
            # inputs = self.input[idx][1]
            # targets = self.input[idx][2]
            #
            # # transform to torch from numpy
            # conditions = torch.from_numpy(conditions)
            # inputs = torch.from_numpy(inputs)
            # targets = torch.from_numpy(targets)
            #
            # return conditions, inputs, targets
            if self.mode == 'train':
                noisy_conditions = self.input[idx][3]  # 2: 10dB, 3: 20dB, 4: 30dB, 5: 40dB
                inputs = self.input[idx][0]
                targets = self.input[idx][1]

                # transform to torch from numpy
                noisy_conditions = torch.from_numpy(noisy_conditions)
                inputs = torch.from_numpy(inputs)
                targets = torch.from_numpy(targets)

                return noisy_conditions, inputs, targets
            if self.mode == 'valid':
                inputs = self.input[idx][0]
                targets = self.input[idx][1]

                # transform to torch from numpy
                inputs = torch.from_numpy(inputs)
                targets = torch.from_numpy(targets)

                return inputs, targets
        else:
            inputs = self.input[idx][0]
            targets = self.input[idx][1]

            # transform to torch from numpy
            inputs = torch.from_numpy(inputs)
            targets = torch.from_numpy(targets)

            return inputs, targets

