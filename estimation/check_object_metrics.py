"""
for checking speech quality with some metrics.

1. PESQ
2. STOI
3. CSIG, CBAK, COVL
"""
import os
from tools_for_estimate import cal_pesq, cal_stoi, composite
from pathlib import Path

# number of files we want to check
flie_num = 1

target_wav = ['.wav']
estimated_wav = ['.wav']

file_directory = '/'

if flie_num == 1:
    pesq = cal_pesq(estimated_wav, target_wav)
    stoi = cal_stoi(estimated_wav, target_wav)
    CSIG, CBAK, CVOL, _ = composite(target_wav[0], estimated_wav[0])

    print('{} is ...'.format(estimated_wav[0]))
    print('PESQ {:.4} | STOI {:.4} | CSIG {:.4} | CBAK {:.4} | CVOL {:.4}'
          .format(pesq, stoi, CSIG, CBAK, CVOL))
else:
    # the list of files in file directory
    if os.path.isdir(file_directory) is False:
        print("[Error] There is no directory '%s'." % file_directory)
        exit()
    else:
        print("Scanning a directory %s " % file_directory)

    # pick target wav from the directory
    target_addr = []
    for path, dir, files in os.walk(file_directory):
        for file in files:
            if file in 'target':
                filepath = Path(path) / file
                target_addr.append(filepath)

    for addr in target_addr:
        estimated_addr = str(addr).replace('target', 'estimated')

        pesq = cal_pesq([estimated_addr], [addr])
        stoi = cal_stoi([estimated_addr], [addr])
        CSIG, CBAK, CVOL, _ = composite(addr, estimated_addr)

        print('{} is ...'.format(estimated_addr))
        print('PESQ {:.4} | STOI {:.4} | CSIG {:.4} | CBAK {:.4} | CVOL {:.4}'
              .format(pesq, stoi, CSIG, CBAK, CVOL))
