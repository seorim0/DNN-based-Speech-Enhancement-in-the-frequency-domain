cmd_list = ['mp3_to_wav', 'noise_resampling']
cmd = ''

if cmd == 'mp3_to_wav':
    from pydub import AudioSegment

    sound = AudioSegment.from_mp3("./data_for_tank/noise/a.mp3")
    sound.export("./data_for_tank/noise/a.wav", format="wav")


elif cmd == 'noise_resampling':
    import os
    from pathlib import Path
    import soundfile
    import data_config as cfg
    import librosa
    import scipy.io.wavfile as wav_write

    dir_name = './data/noise/'

    if os.path.isdir(dir_name) is False:
        print("[Error] There is no directory '%s'." % dir_name)
        exit()
    else:
        print("Scanning a directory %s " % dir_name)

    noise_list = []
    for path, dirs, files in os.walk(dir_name):
        for file in files:
            if file.endswith(".wav"):
                filepath = Path(path) / file
                noise_list.append(filepath)

    for addr in noise_list:
        wav, fs = soundfile.read(addr)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if fs != cfg.fs:
            wav_speech = librosa.resample(wav, fs, cfg.fs)
        wav_write.write(addr, cfg.fs, wav)
