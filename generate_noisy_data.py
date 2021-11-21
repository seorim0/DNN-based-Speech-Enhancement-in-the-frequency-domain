"""
generate noisy data with various noise files
"""
import os
import sys
import numpy as np
import scipy.io.wavfile as wav
import librosa
from pathlib import Path
import soundfile

#######################################################################
#                         data info setting                           #
#######################################################################
# USE THIS, OR SYS.ARGVS
# mode = 'train'  # train / validation / test
# snr_set = [0, 5]
# fs = 16000

#######################################################################
#                                main                                 #
#######################################################################
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
    argvs = sys.argv[1:]
    if len(argvs) != 3:
        print('Error: Invalid input arguments')
        print('\t Usage: python generate_noisy_data.py [mode] [snr] [fs]')
        print("\t\t [mode]: 'train', 'validation'")
        print("\t\t [snr]: '0', '0, 5', ...'")
        print("\t\t [fs]: '16000', ...")
        exit()
    mode = argvs[0]
    snr_set = argvs[1].split(',')
    fs = int(argvs[2])

    # Set speech and noise directory.
    speech_dir = Path("./")

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
    log_file_name = Path("./log_generate_data_" + mode + ".txt")
    f = open(log_file_name, 'w')

    if mode == 'train':
        # Make a noise file list
        noise_subset_dir = speech_dir / 'train' / 'noise'
        list_noise_files = scan_directory(noise_subset_dir)
        for snr_in_db in snr_set:
            for addr_speech in list_speech_files:
                # Load speech waveform and its sampling frequency.
                wav_speech, read_fs = soundfile.read(addr_speech)
                if read_fs != fs:
                    wav_speech = librosa.resample(wav_speech, read_fs, fs)

                # Select a noise component randomly, and read it.
                nidx = np.random.randint(0, len(list_noise_files))
                addr_noise = list_noise_files[nidx]
                wav_noise, read_fs = soundfile.read(addr_noise)
                if wav_noise.ndim > 1:
                    wav_noise = wav_noise.mean(axis=1)
                if read_fs != fs:
                    wav_noise = librosa.resample(wav_noise, read_fs, fs)

                # Generate noisy speech by mixing speech and noise components.
                wav_noisy = generate_noisy_wav(wav_speech, wav_noise, int(snr_in_db))
                noisy_name = Path(addr_speech).name[:-4] +'_' + Path(addr_noise).name[:-4] + '_' + str(
                                  int(snr_in_db)) + '.wav'
                addr_noisy = speech_mode_noisy_dir / noisy_name
                wav.write(addr_noisy, fs, wav_noisy)

                # Display progress.
                print('%s > %s' % (addr_speech, addr_noisy))
                f.write('%s\t%s\t%s\t%d dB\n' % (addr_noisy, addr_speech, addr_noise, int(snr_in_db)))

    elif mode == 'validation':
        # Make a noise file list for validation.
        noise_subset_dir = speech_dir / 'train' / 'noise'
        list_noise_files = scan_directory(noise_subset_dir)

        for addr_speech in list_speech_files:
            # Load speech waveform and its sampling frequency.
            wav_speech, read_fs = soundfile.read(addr_speech)
            if read_fs != fs:
                wav_speech = librosa.resample(wav_speech, read_fs, fs)

            # Select a noise component randomly, and read it.
            nidx = np.random.randint(0, len(list_noise_files))
            addr_noise = list_noise_files[nidx]
            wav_noise, read_fs = soundfile.read(addr_noise)
            if wav_noise.ndim > 1:
                wav_noise = wav_noise.mean(axis=1)
            if read_fs != fs:
                wav_noise = librosa.resample(wav_noise, read_fs, fs)

            # Select an SNR randomly.
            ridx_snr = np.random.randint(0, len(snr_set))
            snr_in_db = int(snr_set[ridx_snr])

            # Generate noisy speech by mixing speech and noise components.
            wav_noisy = generate_noisy_wav(wav_speech, wav_noise, snr_in_db)

            # Write the generated noisy speech into a file.
            noisy_name = Path(addr_speech).name[:-4] + '_' + Path(addr_noise).name[:-4] + '_' + str(
                              snr_in_db) + '.wav'
            addr_noisy = speech_mode_noisy_dir / noisy_name
            wav.write(addr_noisy, fs, wav_noisy)

            # Display progress.
            print('%s > %s' % (addr_speech, addr_noisy))
            f.write('%s\t%s\t%s\t%d dB\n' % (addr_noisy, addr_speech, addr_noise, snr_in_db))
    f.close()


if __name__ == '__main__':
    main()
