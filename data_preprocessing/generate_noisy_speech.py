"""
generate noisy data with one or various noise files
"""
import os
import numpy as np
import scipy.io.wavfile as wav
import librosa
import random
import data_config as cfg
from pathlib import Path
import soundfile


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
                addr.append(filepath)
    return addr


# Generate noisy data given speech, noise, and target SNR.
def generate_noisy_wav(wav_speech, wav_noise, snr):
    # Obtain the length of speech and noise components.
    len_speech = len(wav_speech)
    len_noise = len(wav_noise)

    # Select noise segment randomly to have same length with speech signal.
    st = np.random.randint(0, len_noise - len_speech)
    ed = st + len_speech
    wav_noise = wav_noise[st:ed]

    # Compute the power of speech and noise after removing DC bias.
    dc_speech = np.mean(wav_speech)
    dc_noise = np.mean(wav_noise)
    pow_speech = np.mean(np.power(wav_speech - dc_speech, 2.0))
    pow_noise = np.mean(np.power(wav_noise - dc_noise, 2.0))

    # Compute the scale factor of noise component depending on the target SNR.
    alpha = np.sqrt(10.0 ** (float(-snr) / 10.0) * pow_speech / (pow_noise + 1e-6))
    noisy_wav = (wav_speech + alpha * wav_noise) * 32768
    noisy_wav = noisy_wav.astype(np.int16)

    return noisy_wav


def main():
    mode = cfg.mode
    # Set speech and noise directory.
    speech_dir = Path("./data/")
    noise_dir = Path("./data/noise/")

    snr_set = cfg.snr_set

    train_noise_subset = cfg.noise_subset

    # Make a speech file list.
    speech_mode_clean_dir = speech_dir / mode / 'clean'
    speech_mode_noisy_dir = speech_dir / mode / 'noisy'
    list_speech_files = scan_directory(speech_mode_clean_dir)

    # Make directories of the mode and noisy data.
    if os.path.isdir(speech_mode_clean_dir) is False:
        os.system('mkdir ' + str(speech_mode_clean_dir))

    if os.path.isdir(speech_mode_noisy_dir) is False:
        os.system('mkdir ' + str(speech_mode_noisy_dir))

    # Define a log file name.
    log_file_name = Path("./data/log_generate_data_" + mode + ".txt")
    f = open(log_file_name, 'w')

    if mode == 'train':
        # Make a noise file list for validation.
        list_noise_files = []
        if len(train_noise_subset) == 0:
            for i in range(9):
                list_tmp = scan_directory(noise_dir)
                list_noise_files.append(list_tmp)
        else:
            for idx in range(len(train_noise_subset)):
                noise_subset_dir = noise_dir / 'train' / train_noise_subset[idx]
                list_tmp = scan_directory(noise_subset_dir)
                list_noise_files.append(list_tmp)

        for snr_in_db in snr_set:

            # ############################################
            ## if want to make small data
            # random.shuffle(list_speech_files)
            # list_speech_files = list_speech_files[:1000]
            # ############################################
            for addr_speech in list_speech_files:
                # Load speech waveform and its sampling frequency.
                wav_speech, fs = soundfile.read(addr_speech)
                if fs != cfg.fs:
                    wav_speech = librosa.resample(wav_speech, fs, cfg.fs)
                # wav_speech = signal.decimate(wav_speech, int(fs / cfg.fs))

                # Select a noise component randomly, and read it.
                nidx = np.random.randint(0, len(list_noise_files))
                ridx_noise_file = np.random.randint(0, len(list_noise_files[nidx]))
                addr_noise = list_noise_files[nidx][ridx_noise_file]
                wav_noise, fs = soundfile.read(addr_noise)
                if wav_noise.ndim > 1:
                    wav_noise = wav_noise.mean(axis=1)
                if fs != cfg.fs:
                    wav_noise = librosa.resample(wav_noise, fs, cfg.fs)
                # wav_noise = signal.decimate(wav_noise, int(fs / cfg.fs))

                # Generate noisy speech by mixing speech and noise components.
                wav_noisy = generate_noisy_wav(wav_speech, wav_noise, snr_in_db)
                # Write the generated noisy speech into a file.

                # noisy_name = str(addr_speech).replace('clean', 'noisy')[:-4] + '_' + str(snr_in_db) + '.wav'
                # addr_noisy = noisy_name
                noisy_name = Path(addr_speech).name[:-4] + '_' + Path(addr_noise).name[:-4] + '_' + str(
                    snr_in_db) + '.wav'
                addr_noisy = speech_mode_noisy_dir / noisy_name
                wav.write(addr_noisy, cfg.fs, wav_noisy)

                # Display progress.
                print('%s > %s' % (addr_speech, addr_noisy))
                f.write('%s\t%s\t%s\t%d dB\n' % (addr_noisy, addr_speech, addr_noise, snr_in_db))

    elif mode == 'validation':
        # Make a noise file list for validation.
        list_noise_files = []
        if len(train_noise_subset) == 0:
            list_tmp = scan_directory(noise_dir)
            list_noise_files.append(list_tmp)
        else:
            for idx in range(len(train_noise_subset)):
                noise_subset_dir = noise_dir / 'train' / train_noise_subset[idx]
                list_tmp = scan_directory(noise_subset_dir)
                list_noise_files.append(list_tmp)

        for addr_speech in list_speech_files:
            # Load speech waveform and its sampling frequency.
            wav_speech, fs = soundfile.read(addr_speech)
            # wav_speech = signal.decimate(wav_speech, int(fs / cfg.fs))

            # Select a noise component randomly, and read it.
            nidx = np.random.randint(0, len(list_noise_files))
            ridx_noise_file = np.random.randint(0, len(list_noise_files[nidx]))
            addr_noise = list_noise_files[nidx][ridx_noise_file]
            wav_noise, fs = soundfile.read(addr_noise)
            if wav_noise.ndim > 1:
                wav_noise = wav_noise.mean(axis=1)
            # wav_noise = signal.decimate(wav_noise, int(fs / cfg.fs))

            # Select an SNR randomly.
            ridx_snr = np.random.randint(0, len(snr_set))
            snr_in_db = snr_set[ridx_snr]

            # Generate noisy speech by mixing speech and noise components.
            wav_noisy = generate_noisy_wav(wav_speech, wav_noise, snr_in_db)

            # Write the generated noisy speech into a file.

            # noisy_name = str(addr_speech).replace('clean', 'noisy')[:-4] + '_' + str(snr_in_db) + '.wav'
            # addr_noisy = noisy_name
            noisy_name = Path(addr_speech).name[:-4] + '_' + Path(addr_noise).name[:-4] + '_' + str(
                snr_in_db) + '.wav'
            addr_noisy = speech_mode_noisy_dir / noisy_name
            wav.write(addr_noisy, cfg.fs, wav_noisy)

            # Display progress.
            print('%s > %s' % (addr_speech, addr_noisy))
            f.write('%s\t%s\t%s\t%d dB\n' % (addr_noisy, addr_speech, addr_noise, snr_in_db))

    elif mode == 'test':
        # Make a noise file list for testing.
        list_noise_files = []
        if len(train_noise_subset) == 0:
            list_tmp = scan_directory(noise_dir)
            list_noise_files.append(list_tmp)
        else:
            for idx in range(len(train_noise_subset)):
                noise_subset_dir = noise_dir / 'test' / train_noise_subset[idx]
                list_tmp = scan_directory(noise_subset_dir)
                list_noise_files.append(list_tmp)

        for didx in snr_set:
            for addr_speech in list_speech_files:
                # Load speech waveform and its sampling frequency.
                wav_speech, fs = librosa.load(addr_speech, cfg.fs)
                if fs != cfg.fs:
                    wav_speech = librosa.resample(wav_speech, fs, cfg.fs)
                # wav_speech = signal.decimate(wav_speech, int(fs / cfg.fs))

                # Select a noise component randomly, and read it.
                nidx = np.random.randint(0, len(list_noise_files))
                ridx_noise_file = np.random.randint(0, len(list_noise_files[nidx]))
                addr_noise = list_noise_files[nidx][ridx_noise_file]
                wav_noise, fs = librosa.load(addr_noise, cfg.fs)
                if wav_noise.ndim > 1:
                    wav_noise = wav_noise.mean(axis=1)
                if fs != cfg.fs:
                    wav_noise = librosa.resample(wav_noise, fs, cfg.fs)
                # wav_noise = signal.decimate(wav_noise, int(fs / cfg.fs))

                # Generate noisy speech by mixing speech and noise components.
                wav_noisy = generate_noisy_wav(wav_speech, wav_noise, didx)

                # Write the generated noisy speech into a file.
                noisy_name = Path(addr_speech).name[:-4] + '_' \
                             + Path(addr_noise).name[:-4] + '_' + str(didx) + '.wav'
                dB = str(didx) + 'dB'
                addr_noisy = speech_dir / mode / 'noisy' / cfg.test_noise_type / dB / noisy_name
                wav.write(addr_noisy, cfg.fs, wav_noisy)

                # Display progress.
                print(cfg.test_noise_type, '%s > %s' % (addr_speech, addr_noisy))
                f.write(cfg.test_noise_type, '%s\t%s\t%s\t%d dB\n' % (addr_noisy, addr_speech, addr_noise, didx))

    f.close()


if __name__ == '__main__':
    main()
