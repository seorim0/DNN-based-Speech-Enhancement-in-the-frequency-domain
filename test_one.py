import time
import os
import numpy as np
import torch
from tools_for_estimate import cal_pesq, cal_stoi, cal_snr_array, composite
from tools_for_model import pam_pw_draw
from model import complex_model
from scipy.io.wavfile import write as wav_write

np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

# expr_num = 'P008'
# dir_to_save = './models/P008_7.12_DCCRN_MSE'
# epoch = 44
#expr_num = 'BIG1'
dir_to_save = './models/C1_B15_MSE'
epoch = 10

# Set device
DEVICE = torch.device('cuda')
# MS_30_speaker_diff_seen_5dB
# MS_3696_unseen_5dB
for i in range(1):
    # test_name = 'MS_3696_seen_{}dB'.format(i*5)
    # test_dataset = np.load('./input/' + test_name + '_test_dataset.npy')
    test_name = 'VALIDATION_TEST'
    test_dataset = np.load('./input/C1+_validation_dataset.npy')

    # Set model
    model = complex_model().to(DEVICE)

    checkpoint = torch.load(dir_to_save + '/chkpt_' + str(epoch) + '.pt')
    model.load_state_dict(checkpoint['model'])
    epoch_start_idx = checkpoint['epoch'] + 1

    # initialize
    batch_num = 0

    avg_pesq = 0
    avg_stoi = 0
    avg_snr = 0
    avg_noisy_pesq = 0
    avg_noisy_stoi = 0
    avg_noisy_snr = 0
    avg_csig = 0
    avg_cbak = 0
    avg_cvol = 0

    all_batch_input = []
    all_batch_target = []
    all_batch_output = []
    all_batch_pesq = []
    all_batch_noisy_pesq = []
    all_batch_stoi = []

    # for record the score each samples
    f_score = open(dir_to_save + '/' + test_name + '_Epoch_{}_SCORES'.format(epoch), 'a')
    print('Test type {}'.format(test_name))
    f_score.write('Test type {}\n'.format(test_name))

    start_time = time.time()
    for inputs, targets in test_dataset:
        batch_num += 1

        # transform to torch from numpy
        inputs = torch.Tensor([inputs])
        targets = torch.Tensor([targets])
        # to cuda
        inputs = inputs.float().to(DEVICE)
        targets = targets.float().to(DEVICE)

        _, _, outputs = model(inputs)

        targets = targets[:,:len(outputs[0])]
        residual_noise_wavs = outputs - targets
        # if batch_num == 3:
        #     pam_pw_draw(inputs, targets, outputs, residual_noise_wavs, None, dir_to_save, 16000, 512, epoch)

        # estimate the output speech with pesq and stoi
        estimated_wavs = outputs.cpu().detach().numpy()
        clean_wavs = targets.cpu().detach().numpy()
        noisy_wavs = inputs.cpu().detach().numpy()

        pesq = cal_pesq(estimated_wavs, clean_wavs)
        noisy_pesq = cal_pesq(noisy_wavs, clean_wavs)
        if len(estimated_wavs[0]) > len(clean_wavs[0]):  # why?
            estimated_wavs[0] = estimated_wavs[:, :len(clean_wavs[0])]
        else:
            clean_wavs = clean_wavs[:, :len(estimated_wavs[0])]
            noisy_wavs = noisy_wavs[:, :len(estimated_wavs[0])]
        stoi = cal_stoi(estimated_wavs, clean_wavs)
        noisy_stoi = cal_stoi(noisy_wavs, clean_wavs)
        snr = cal_snr_array(estimated_wavs, clean_wavs)
        noisy_snr = cal_snr_array(noisy_wavs, clean_wavs)

        # reshape for sum
        pesq = np.reshape(pesq, (1, -1))
        stoi = np.reshape(stoi, (1, -1))
        snr = np.reshape(snr, (1, -1))

        noisy_pesq = np.reshape(noisy_pesq, (1, -1))
        noisy_stoi = np.reshape(noisy_stoi, (1, -1))
        noisy_snr = np.reshape(noisy_snr, (1, -1))

        f_score.write('PESQ: REF {:.6} EST {:.6} | STOI: REF {:.6} EST {:.6} | SNR: REF {:.6} EST {:.6}\n'
                      .format(noisy_pesq[0][0], pesq[0][0], noisy_stoi[0][0], stoi[0][0], noisy_snr[0][0], snr[0][0]))

        # all batch data array
        all_batch_input.extend(inputs.cpu().detach().numpy())
        all_batch_target.extend(targets.cpu().detach().numpy())
        all_batch_output.extend(outputs.cpu().detach().numpy())
        all_batch_pesq.extend(pesq[0])
        all_batch_stoi.extend(stoi[0])

        all_batch_noisy_pesq.extend(noisy_pesq[0])

        avg_pesq += sum(pesq[0]) / len(inputs)
        avg_stoi += sum(stoi[0]) / len(inputs)
        avg_snr += sum(snr[0]) / len(inputs)

        avg_noisy_pesq += sum(noisy_pesq[0]) / len(inputs)
        avg_noisy_stoi += sum(noisy_stoi[0]) / len(inputs)
        avg_noisy_snr += sum(noisy_snr[0]) / len(inputs)
        if batch_num % 10 == 0:
            print('{}/{} done! takes {} seconds'.format(batch_num, len(test_dataset), time.time() - start_time))

    # # make the file directory to save the samples
    # if not os.path.exists(dir_to_save + '/output'):
    #     os.mkdir(dir_to_save + '/output')
    # if not os.path.exists(dir_to_save + '/output/{}'.format(test_name)):
    #     os.mkdir(dir_to_save + '/output/{}'.format(test_name))
    #
    # estfile_path = dir_to_save + '/output/{}/'.format(test_name)
    # for m in range(len(all_batch_output)):
    #     est_file_name = '{}_{}_est_{:.5}.wav'.format(m + 1, expr_num, all_batch_pesq[m])
    #
    #     est_wav = all_batch_output[m]
    #     wav_write(estfile_path + est_file_name, 16000, est_wav)
    #
    #     noisy_file_name = '{}_noisy_{:.5}.wav'.format(m + 1, all_batch_noisy_pesq[m])
    #     noisy_wav = all_batch_input[m]
    #     wav_write(estfile_path + noisy_file_name, 16000, noisy_wav)
    #
    #     clean_file_name = '{}_clean.wav'.format(m + 1)
    #     clean_wav = all_batch_target[m]
    #     wav_write(estfile_path + clean_file_name, 16000, clean_wav)

        # CSIG, CBAK, CVOL, _ = composite(estfile_path + clean_file_name, estfile_path + est_file_name)
        # avg_csig += CSIG
        # avg_cbak += CBAK
        # avg_cvol += CVOL

        # # pesq: 0.1 better / stoi: 0.01 better
        # f_score.write('PESQ {:.6f} | STOI {:.6f} | CSIG {:.6f} | CBAK {:.6f} | CVOL {:.6f}\n'
        #               .format(all_batch_pesq[m], all_batch_stoi[m], CSIG, CBAK, CVOL))

    avg_pesq /= batch_num
    avg_stoi /= batch_num
    avg_snr /= batch_num
    avg_noisy_pesq /= batch_num
    avg_noisy_stoi /= batch_num
    avg_noisy_snr /= batch_num
    avg_csig /= batch_num
    avg_cbak /= batch_num
    avg_cvol /= batch_num

    print('PESQ: REF {:.6} EST {:.6} | STOI: REF {:.6} EST {:.6} | SNR: REF {:.6} EST {:.6}'
          .format(avg_noisy_pesq, avg_pesq, avg_noisy_stoi, avg_stoi, avg_noisy_snr, avg_snr))
    # print('REF CSIG {:.6f} | CBAK {:.6f} | COVL {:.6f}'.format(noisy_csig, noisy_cbak, noisy_cvol))
    # print('    CSIG {:.6f} | CBAK {:.6f} | COVL {:.6f}'.format(test_csig, test_cbak, test_cvol))
    f_score.write('\n\n Avg.\n')
    f_score.write('PESQ: REF {:.6} EST {:.6} | STOI: REF {:.6} EST {:.6} | SNR: REF {:.6} EST {:.6}\n'
                  .format(avg_noisy_pesq, avg_pesq, avg_noisy_stoi, avg_stoi, avg_noisy_snr, avg_snr))

    f_score.close()
