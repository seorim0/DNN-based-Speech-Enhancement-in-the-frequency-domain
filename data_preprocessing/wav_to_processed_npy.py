"""
Read wav files, normalize them, and convert to numpy data format
"""
"""
Read wav speech files and transform the shape of data for input to the model

data shape: [data number, sampling frequency * seconds]
            [18480, 8000 * 3]
data scope: [-1 1]

Dataset: [input(noisy), target(clean)]
         = [18480, 2, 8000*3]
"""
import os
import soundfile
import librosa
import numpy as np
from pathlib import Path
from tools_for_model import normalize_dataset
import data_config as cfg


# noisy - clean pair
def scan_directory(dir_name):
    """Scan directory and save address of clean/noisy wav data.
    Args:
        dir_name: directroy name to scan

    Returns:
        addr: all address list of clean/noisy wave data in subdirectory
    """
    if os.path.isdir(dir_name) is False:
        print("[Error] There is no directory '%s'." % dir_name)
        exit()
    else:
        print("Scanning a directory %s " % dir_name)

    addr = []
    for subdir, dirs, files in os.walk(dir_name):
        for file in files:
            if file.endswith(".wav"):
                filepath = Path(subdir) / file
                addr_noisy = filepath
                addr_clean = str(filepath).replace('noisy', 'clean')
                # 1st '_'
                idx_1st = addr_clean.find('_')
                # 2nd '_'
                idx_2nd = addr_clean[idx_1st + 1:].find('_')
                # 3rd '_'
                idx_3rd = addr_clean[idx_1st + 1 + idx_2nd + 1:].find('_')
                # 4th '_'
                idx_4th = addr_clean[idx_1st + 1 + idx_2nd + 1 + idx_3rd + 1:].find('_')
                if not cfg.pam:
                    addr_clean = addr_clean[:idx_1st + 1 + idx_2nd + 1 + idx_3rd + 1 + idx_4th] + '.wav'
                    addr.append([addr_noisy, addr_clean])
                else:
                    # 1st '/'
                    idx2_1st = addr_clean.find('/')
                    # 2nd '/'
                    idx2_2nd = addr_clean[idx2_1st + 1:].find('/')
                    # 3rd '/'
                    idx2_3rd = addr_clean[idx2_1st + 1 + idx2_2nd + 1:].find('/')

                    file_name = addr_clean[idx2_1st + 1 + idx2_2nd + 1 + idx2_3rd + 1: idx_1st + 1 + idx_2nd + 1 + idx_3rd + 1 + idx_4th] + '.wav'
                    addr_clean = addr_clean[:idx_1st + 1 + idx_2nd + 1 + idx_3rd + 1 + idx_4th] + '.wav'
                    addr.append([addr_noisy, addr_clean, file_name])
    return addr


def scan_directory_for_test(clean_dir_name, noisy_dir_name):
    """Scan directory and save address of clean/noisy wav data.
    Args:
        dir_name: directroy name to scan

    Returns:
        addr: all address list of clean/noisy wave data in subdirectory
    """
    if os.path.isdir(noisy_dir_name) is False:
        print("[Error] There is no directory '%s'." % noisy_dir_name)
        exit()
    else:
        print("Scanning a directory %s " % noisy_dir_name)

    addr = []
    for subdir, dirs, files in os.walk(noisy_dir_name):
        for file in files:
            if file.endswith(".wav"):
                filepath = Path(subdir) / file
                addr_noisy = str(filepath)
                # 1st '_'
                idx_1st = addr_noisy.find('_')
                # 2nd '_'
                idx_2nd = addr_noisy[idx_1st + 1:].find('_')
                # 3rd '_'
                idx_3rd = addr_noisy[idx_1st + 1 + idx_2nd + 1:].find('_')
                # 4th '_'
                idx_4th = addr_noisy[idx_1st + 1 + idx_2nd + 1 + idx_3rd + 1:].find('_')
            #########
                file_name = addr_noisy[idx_1st - 4: idx_1st + 1 + idx_2nd + 1 + idx_3rd + 1 + idx_4th] + '.wav'
                if not cfg.pam:
                    addr_clean = str(clean_dir_name) + '/' + str(file_name)
                else:
            ##########
                    # 1st '/'
                    idx2_1st = addr_clean.find('/')
                    # 2nd '/'
                    idx2_2nd = addr_clean[idx2_1st + 1:].find('/')
                    # 3rd '/'
                    idx2_3rd = addr_clean[idx2_1st + 1 + idx2_2nd + 1:].find('/')

                    file_name = addr_clean[
                                idx2_1st + 1 + idx2_2nd + 1 + idx2_3rd: idx_1st + 1 + idx_2nd + 1 + idx_3rd + 1 + idx_4th] + '.wav'
                    addr_clean = file_name
                addr.append([addr_noisy, addr_clean])
    return addr


# normalize [-1 1]
def normalize_pam_dataset(dataset):
    for i in range(len(dataset)):
        noisy_max = np.max(abs(dataset[i][0]))
        dataset[i][0] = dataset[i][0] / noisy_max

        clean_max = np.max(abs(dataset[i][1][0]))
        dataset[i][1][0] = dataset[i][1][0] / clean_max
    return dataset


# setup
mode = cfg.mode
d_name = cfg.data_name

if not cfg.pam:
    if mode != 'test':
        noisy_speech = Path('./data/' + mode + '/noisy/')

        noisy_speech_list = scan_directory(noisy_speech)

        # initialize
        speech_dataset = []

        # read wav files
        for addr_speech in noisy_speech_list:
            noisy_speech, fs = soundfile.read(addr_speech[0])
            if fs != cfg.fs:
                noisy_speech = librosa.resample(noisy_speech, fs, cfg.fs)

            clean_speech, fs = soundfile.read(addr_speech[1])
            if fs != cfg.fs:
                clean_speech = librosa.resample(clean_speech, fs, cfg.fs)
            speech_dataset.append([noisy_speech, clean_speech])

        # normalization [-1 1]
        speech_dataset = normalize_dataset(speech_dataset)

        # padding for short wave file
        for k in range(len(speech_dataset)):
            if len(speech_dataset[k][0]) / cfg.fs < 3:
                pad_len = cfg.fs * 3 - len(speech_dataset[k][0])
                speech_dataset[k][0] = np.concatenate((speech_dataset[k][0], np.zeros(pad_len)), 0)
                speech_dataset[k][1] = np.concatenate((speech_dataset[k][1], np.zeros(pad_len)), 0)
            else:
                speech_dataset[k][0] = speech_dataset[k][0][: cfg.fs * 3]  # 0 ~ 3 sec
                speech_dataset[k][1] = speech_dataset[k][1][: cfg.fs * 3]  # 0 ~ 3 sec

        # save to numpy
        print('Noisy data number {}'.format(len(speech_dataset)))
        print('Save dataset...')
        np.save('./' + mode + '_dataset_' + d_name + '.npy', speech_dataset)
        print('Complete.')
    else:
        snr_per_speech_dataset = []
        test_speech_dataset = []

        noise_type = ['seen/']  # seen / unseen
        snr = ['0dB/', '5dB/', '10dB/', '15dB', '20dB']
        for n_type in noise_type:
            for snr_v in snr:
                clean_speech = Path('./data/' + mode + '/clean/')
                if not os.path.exists('./data/' + mode + '/noisy/' + n_type + snr_v):
                    os.mkdir('./data/' + mode + '/noisy/' + n_type + snr_v)
                noisy_speech = Path('./data/' + mode + '/noisy/' + n_type + snr_v)

                noisy_speech_list = scan_directory_for_test(clean_speech, noisy_speech)

                # initialize
                speech_dataset = []

                # read wav files
                for addr_speech in noisy_speech_list:
                    noisy_speech, fs = soundfile.read(addr_speech[0])
                    if fs != cfg.fs:
                        noisy_speech = librosa.resample(noisy_speech, fs, cfg.fs)

                    clean_speech, fs = soundfile.read(addr_speech[1])
                    if fs != cfg.fs:
                        clean_speech = librosa.resample(clean_speech, fs, cfg.fs)
                    speech_dataset.append([noisy_speech, clean_speech])

                # normalization [-1 1]
                speech_dataset = normalize_dataset(speech_dataset)

                # padding for short wave file
                if cfg.padding == True:
                    for k in range(len(speech_dataset)):
                        if len(speech_dataset[k][0]) / cfg.fs < 3:
                            pad_len = cfg.fs * 3 - len(speech_dataset[k][0])
                            speech_dataset[k][0] = np.concatenate((speech_dataset[k][0], np.zeros(pad_len)), 0)
                            speech_dataset[k][1] = np.concatenate((speech_dataset[k][1], np.zeros(pad_len)), 0)
                        else:
                            speech_dataset[k][0] = speech_dataset[k][0][: cfg.fs * 3]  # 0 ~ 3 sec
                            speech_dataset[k][1] = speech_dataset[k][1][: cfg.fs * 3]  # 0 ~ 3 sec

                snr_per_speech_dataset.append(speech_dataset)

            # [seen, unseen] > [-10dB, .., 10dB] > [noisy, clean]
            test_speech_dataset.append(snr_per_speech_dataset)
            snr_per_speech_dataset = []
        # save to numpy
        print('Noisy data number {}'.format(len(speech_dataset)))
        print('Save test dataset...')
        np.save('./' + mode + '_dataset' + d_name + '.npy', test_speech_dataset)
        print('Complete.')
else:
    np_load_old = np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

    if mode != 'test':
        noisy_speech = Path('./data/' + mode + '/noisy/')

        noisy_speech_list = scan_directory(noisy_speech)

        pam = np.load('../input/' + cfg.data_name + '_' + mode + '_pam.npy')

        # initialize
        speech_dataset = []

        # read wav files
        for addr_noisy, addr_clean, clean_file_name in noisy_speech_list:
            noisy_speech, fs = soundfile.read(addr_noisy)
            if fs != cfg.fs:
                noisy_speech = librosa.resample(noisy_speech, fs, cfg.fs)


            pam_clean_name = pam[:,0]
            # corr_index = np.where(pam_clean_name == Path(clean_file_name))
            # corr_index = max(corr_index[0])
            for i in range(len(pam_clean_name)):
                if pam_clean_name[i] == clean_file_name:
                    corr_index = i
                    break

            clean_speech, fs = soundfile.read(addr_clean)
            if fs != cfg.fs:
                clean_speech = librosa.resample(clean_speech, fs, cfg.fs)
            clean_pam = pam[corr_index][1]

            speech_dataset.append([noisy_speech, [clean_speech, clean_pam]])

        # normalization [-1 1]
        speech_dataset = normalize_pam_dataset(speech_dataset)

        # padding for short wave file
        for k in range(len(speech_dataset)):
            if len(speech_dataset[k][0]) / cfg.fs < 3:
                pad_len = cfg.fs * 3 - len(speech_dataset[k][0])
                speech_dataset[k][0] = np.concatenate((speech_dataset[k][0], np.zeros(pad_len)), 0)
                speech_dataset[k][1][0] = np.concatenate((speech_dataset[k][1][0], np.zeros(pad_len)), 0)
            else:
                speech_dataset[k][0] = speech_dataset[k][0][: cfg.fs * 3]  # 0 ~ 3 sec
                speech_dataset[k][1][0] = speech_dataset[k][1][0][: cfg.fs * 3]  # 0 ~ 3 sec

        # save to numpy
        print('Noisy data number {}'.format(len(speech_dataset)))
        print('Save dataset...')
        np.save('../input/PAM_'+ cfg.data_name + mode + '_dataset.npy', speech_dataset)
        print('Complete.')
    # else:
    #     pam = np.load('../input/' + cfg.data_name + '_' + mode + '_pam.npy')
    # 
    #     snr_per_speech_dataset = []
    #     test_speech_dataset = []
    # 
    #     noise_type = ['seen/']  # seen / unseen
    #     snr = ['0dB/', '5dB/', '10dB/', '15dB', '20dB']
    #     for n_type in range(len(noise_type)):
    #         for snr_v in range(len(snr)):
    #             clean_speech = Path('./data/' + mode + '/clean/')
    #             noisy_speech = Path('./data/' + mode + '/noisy/' + noise_type[n_type] + snr[snr_v])
    # 
    #             noisy_speech_list = scan_directory_for_test(clean_speech, noisy_speech)
    # 
    #             # initialize
    #             speech_dataset = []
    # 
    #             # read wav files
    #             for addr_speech in noisy_speech_list:
    #                 noisy_speech, fs = soundfile.read(addr_speech[0])
    #                 if fs != cfg.fs:
    #                     noisy_speech = librosa.resample(noisy_speech, fs, cfg.fs)
    # 
    #                 clean_file_name = addr_speech[1]
    #                 corr_index = np.where(pam[n_type][snr_v][:, 0] == Path(clean_file_name))
    #                 corr_index = max(corr_index[0])
    # 
    #                 clean_pam = pam[n_type][snr_v][corr_index][0]
    #                 clean_speech = pam[n_type][snr_v][corr_index][1]
    # 
    #                 speech_dataset.append([noisy_speech, [clean_speech, clean_pam]])
    # 
    #             # normalization [-1 1]
    #             speech_dataset = normalize_pam_dataset(speech_dataset)
    # 
    #             # padding for short wave file
    #             if cfg.padding == True:
    #                 for k in range(len(speech_dataset)):
    #                     if len(speech_dataset[k][0]) / cfg.fs < 3:
    #                         pad_len = cfg.fs * 3 - len(speech_dataset[k][0])
    #                         speech_dataset[k][0] = np.concatenate((speech_dataset[k][0], np.zeros(pad_len)), 0)
    #                     else:
    #                         speech_dataset[k][0] = speech_dataset[k][0][: cfg.fs * 3]  # 0 ~ 3 sec
    # 
    #             snr_per_speech_dataset.append(speech_dataset)
    # 
    #         # [seen, unseen] > [-10dB, .., 10dB] > [noisy, clean]
    #         test_speech_dataset.append(snr_per_speech_dataset)
    #         snr_per_speech_dataset = []
    #     # save to numpy
    #     print('Noisy data number {}'.format(len(speech_dataset)))
    #     print('Save test dataset...')
    #     np.save('./' + mode + '_dataset' + d_name + '.npy', test_speech_dataset)
    #     print('Complete.')
