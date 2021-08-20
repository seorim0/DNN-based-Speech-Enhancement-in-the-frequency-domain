"""
Where the model is actually trained and validated
"""

import torch
import numpy as np
import tools_for_model as tools
import config as cfg
from tools_for_estimate import cal_pesq, cal_stoi
from tools_for_loss import L1Loss
from tools_for_model import pam_pw_draw
import torch.nn.functional as F
from scipy.io.wavfile import write as wav_write
import matplotlib.pyplot as plt


#######################################################################
#                             For train                               #
#######################################################################
def model_train(model, optimizer, train_loader, direct, DEVICE):
    # initialization
    train_loss = 0
    train_end_loss = 0
    train_mid_loss = 0
    train_main_loss = 0
    train_perceptual_loss = 0
    batch_num = 0

    # arr = []
    # train
    model.train()
    if cfg.latent_vector_loss:
        for inputs, targets in tools.Bar(train_loader):
            batch_num += 1

            # to cuda
            inputs = inputs.float().to(DEVICE)
            targets = targets.float().to(DEVICE)

            targets_mid, outputs_mid, outputs = model(targets, inputs, direct_mapping=direct)

            main_loss = F.mse_loss(outputs, targets, reduction='mean')
            mid_loss = F.mse_loss(outputs_mid, targets_mid, reduction='mean')

            # # min-max normalization ()
            # mid_loss -= 2.3535e-12
            # mid_loss /= (2.5166e-09 - 2.3535e-12)
            # arr.append(mid_loss.item())
            # 이렇게 하면 GD의 의미가 없어짐
            # # for min-max normalization  [-1~1]
            # m = mid_loss.clone()
            # m = m.permute(0, 3, 1, 2)
            # m = m.view(cfg.batch, 483, -1)
            # mid_loss = mid_loss.permute(0, 3, 1, 2)
            # mid_loss = mid_loss.view(cfg.batch, 483, -1)
            #
            # p_max = torch.max(abs(m), 2)
            # p_min = torch.min(abs(m), 2)
            #
            # p_max = p_max[0].unsqueeze(2)
            # p_min = p_min[0].unsqueeze(2)
            #
            # mid_loss -= p_min
            # mid_loss /= (p_max - p_min)
            # mid_loss = torch.mean(mid_loss)

            r1 = 1
            r2 = 1
            r = r1 + r2
            loss = (r1 * main_loss + r2 * mid_loss) / r
            # if you want to check the scale of the loss
            print('main loss: {:.4} mid loss: {:.4}'.format(r1 * main_loss, r2 * mid_loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss
            train_end_loss += r1 * main_loss
            train_mid_loss += r2 * mid_loss
        # np.save('./mid_vector.npy', arr)
        # p_max = np.max(arr)
        # p_min = np.min(arr)
        # print(p_max)
        # print(p_min)

        train_loss /= batch_num
        train_end_loss /= batch_num
        train_mid_loss /= batch_num

        return train_loss, train_end_loss, train_mid_loss
    else:
        if cfg.perceptual == 'PAM':
            for inputs, targets, GMT in tools.Bar(train_loader):
                batch_num += 1

                # to cuda
                inputs = inputs.float().to(DEVICE)
                targets = targets.float().to(DEVICE)
                GMT = GMT.float().to(DEVICE)

                _, _, outputs = model(inputs, direct_mapping=direct)
                main_loss = model.loss(outputs, targets)
                perceptual_loss = model.loss(outputs, targets, GMT=GMT, perceptual=True)

                # the constraint ratio
                r1 = 10e8 * 6
                r2 = 1
                r3 = r1 + r2
                loss = (r1 * main_loss + r2 * perceptual_loss) / r3
                # print('M {:.4}  P {:.4}'.format(r1 * main_loss, r2 * perceptual_loss))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss
                train_main_loss += r1 * main_loss
                train_perceptual_loss += r2 * perceptual_loss
            train_loss /= batch_num
            train_main_loss /= batch_num
            train_perceptual_loss /= batch_num

            return train_loss, train_main_loss, train_perceptual_loss
        elif cfg.perceptual == 'LMS' or cfg.perceptual == 'PMSQE':
            for inputs, targets in tools.Bar(train_loader):
                batch_num += 1

                # to cuda
                inputs = inputs.float().to(DEVICE)
                targets = targets.float().to(DEVICE)

                real_spec, img_spec, outputs = model(inputs, direct_mapping=direct)
                main_loss = model.loss(outputs, targets)
                perceptual_loss = model.loss(outputs, targets, real_spec, img_spec, perceptual=True)

                # the constraint ratio
                r1 = 1
                r2 = 1
                r3 = r1 + r2
                loss = (r1 * main_loss + r2 * perceptual_loss) / r3

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss
                train_main_loss += r1 * main_loss
                train_perceptual_loss += r2 * perceptual_loss
            train_loss /= batch_num
            train_main_loss /= batch_num
            train_perceptual_loss /= batch_num

            return train_loss, train_main_loss, train_perceptual_loss
        else:  # Not use perceptual loss
            for inputs, targets in tools.Bar(train_loader):
                batch_num += 1

                # to cuda
                inputs = inputs.float().to(DEVICE)
                targets = targets.float().to(DEVICE)

                # if direct:
                #     _, _, outputs, targets = model(inputs, targets, direct_mapping=direct)
                # else:
                #     _, _, outputs = model(inputs, direct_mapping=direct)
                _, _, outputs = model(inputs, direct_mapping=direct)
                loss = model.loss(outputs, targets)
                # # if you want to check the scale of the loss
                # print('loss: {:.4}'.format(loss))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss
            train_loss /= batch_num

            return train_loss  ##


def model_direct_train(model, optimizer, train_loader, DEVICE):
    # initialization
    train_loss = 0
    train_end_loss = 0
    train_mid_loss = 0
    train_main_loss = 0
    train_perceptual_loss = 0
    batch_num = 0

    # arr = []
    # train
    model.train()
    if cfg.latent_vector_loss:
        for inputs, targets in tools.Bar(train_loader):
            batch_num += 1

            # to cuda
            inputs = inputs.float().to(DEVICE)
            targets = targets.float().to(DEVICE)

            targets_mid, outputs_mid, outputs = model(targets, inputs, direct_mapping=direct)

            main_loss = F.mse_loss(outputs, targets, reduction='mean')
            mid_loss = F.mse_loss(outputs_mid, targets_mid, reduction='mean')

            # # min-max normalization ()
            # mid_loss -= 2.3535e-12
            # mid_loss /= (2.5166e-09 - 2.3535e-12)
            # arr.append(mid_loss.item())
            # 이렇게 하면 GD의 의미가 없어짐
            # # for min-max normalization  [-1~1]
            # m = mid_loss.clone()
            # m = m.permute(0, 3, 1, 2)
            # m = m.view(cfg.batch, 483, -1)
            # mid_loss = mid_loss.permute(0, 3, 1, 2)
            # mid_loss = mid_loss.view(cfg.batch, 483, -1)
            #
            # p_max = torch.max(abs(m), 2)
            # p_min = torch.min(abs(m), 2)
            #
            # p_max = p_max[0].unsqueeze(2)
            # p_min = p_min[0].unsqueeze(2)
            #
            # mid_loss -= p_min
            # mid_loss /= (p_max - p_min)
            # mid_loss = torch.mean(mid_loss)

            r1 = 1
            r2 = 1
            r = r1 + r2
            loss = (r1 * main_loss + r2 * mid_loss) / r
            # if you want to check the scale of the loss
            print('main loss: {:.4} mid loss: {:.4}'.format(r1 * main_loss, r2 * mid_loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss
            train_end_loss += r1 * main_loss
            train_mid_loss += r2 * mid_loss
        # np.save('./mid_vector.npy', arr)
        # p_max = np.max(arr)
        # p_min = np.min(arr)
        # print(p_max)
        # print(p_min)

        train_loss /= batch_num
        train_end_loss /= batch_num
        train_mid_loss /= batch_num

        return train_loss, train_end_loss, train_mid_loss
    else:
        if cfg.perceptual == 'PAM':
            for inputs, targets, GMT in tools.Bar(train_loader):
                batch_num += 1

                # to cuda
                inputs = inputs.float().to(DEVICE)
                targets = targets.float().to(DEVICE)
                GMT = GMT.float().to(DEVICE)

                _, _, outputs = model(inputs, direct_mapping=direct)
                main_loss = model.loss(outputs, targets)
                perceptual_loss = model.loss(outputs, targets, GMT=GMT, perceptual=True)

                # the constraint ratio
                r1 = 10e8 * 6
                r2 = 1
                r3 = r1 + r2
                loss = (r1 * main_loss + r2 * perceptual_loss) / r3
                # print('M {:.4}  P {:.4}'.format(r1 * main_loss, r2 * perceptual_loss))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss
                train_main_loss += r1 * main_loss
                train_perceptual_loss += r2 * perceptual_loss
            train_loss /= batch_num
            train_main_loss /= batch_num
            train_perceptual_loss /= batch_num

            return train_loss, train_main_loss, train_perceptual_loss
        elif cfg.perceptual == 'LMS' or cfg.perceptual == 'PMSQE':
            for inputs, targets in tools.Bar(train_loader):
                batch_num += 1

                # to cuda
                inputs = inputs.float().to(DEVICE)
                targets = targets.float().to(DEVICE)

                real_spec, img_spec, outputs = model(inputs, direct_mapping=direct)
                main_loss = model.loss(outputs, targets)
                perceptual_loss = model.loss(outputs, targets, real_spec, img_spec, perceptual=True)

                # the constraint ratio
                r1 = 1
                r2 = 1
                r3 = r1 + r2
                loss = (r1 * main_loss + r2 * perceptual_loss) / r3

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss
                train_main_loss += r1 * main_loss
                train_perceptual_loss += r2 * perceptual_loss
            train_loss /= batch_num
            train_main_loss /= batch_num
            train_perceptual_loss /= batch_num

            return train_loss, train_main_loss, train_perceptual_loss
        else:  # Not use perceptual loss
            for inputs, targets in tools.Bar(train_loader):
                batch_num += 1

                # to cuda
                inputs = inputs.float().to(DEVICE)
                targets = targets.float().to(DEVICE)

                # if direct:
                #     _, _, outputs, targets = model(inputs, targets, direct_mapping=direct)
                # else:
                #     _, _, outputs = model(inputs, direct_mapping=direct)
                output_real, target_real, output_imag, target_imag, _ = model(inputs, targets)
                real_loss = model.loss(output_real, target_real)
                imag_loss = model.loss(output_imag, target_imag)
                loss = (real_loss + imag_loss) / 2

                # # if you want to check the scale of the loss
                # print('loss: {:.4}'.format(loss))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss
            train_loss /= batch_num

            return train_loss  ##


def model_conditional_train(model, optimizer, train_loader, direct, DEVICE):
    # initialization
    train_loss = 0
    batch_num = 0

    # train
    model.train()
    for conditions, inputs, targets in tools.Bar(train_loader):
        batch_num += 1

        # to cuda
        conditions = inputs.float().to(DEVICE)
        inputs = inputs.float().to(DEVICE)
        targets = targets.float().to(DEVICE)

        _, _, outputs = model(conditions, inputs, direct_mapping=direct)

        loss = model.loss(outputs, targets)
        # # if you want to check the scale of the loss
        # print('loss: {:.4}'.format(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss
    train_loss /= batch_num

    return train_loss


def cycle_model_train(N2C, C2N, optimizer, train_loader, direct, DEVICE):
    # initialization
    train_loss = 0
    train_main_loss = 0
    train_C2N_NL1_loss = 0
    train_N2C_CL1_loss = 0
    batch_num = 0

    N2C.train()
    C2N.train()
    for inputs, targets in tools.Bar(train_loader):
        batch_num += 1

        # to cuda
        inputs = inputs.float().to(DEVICE)
        targets = targets.float().to(DEVICE)

        _, _, estimated_clean_outputs = N2C(inputs, direct_mapping=direct)
        _, _, fake_noisy_outputs = C2N(estimated_clean_outputs, direct_mapping=True)

        _, _, estimated_noisy_outputs = C2N(targets, direct_mapping=True)
        _, _, fake_clean_outputs = N2C(estimated_noisy_outputs, direct_mapping=direct)

        main_loss = N2C.loss(estimated_clean_outputs, targets)

        C2N_NL1_loss = L1Loss(fake_noisy_outputs, inputs)
        N2C_CL1_loss = L1Loss(fake_clean_outputs, targets)

        # constraint ratio
        r1 = 150
        r2 = 1
        r3 = 1
        r = r1 + r2 + r3

        # # if you want to check the scale of the loss
        # print('M: {:.6} C2N: {:.6} N2C {:.6}'.format(r1 * main_loss, r2 * C2N_NL1_loss, r3 * N2C_CL1_loss))
        loss = (r1 * main_loss + r2 * C2N_NL1_loss + r3 * N2C_CL1_loss) / r

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss
        train_main_loss += r1 * main_loss
        train_C2N_NL1_loss += r2 * C2N_NL1_loss
        train_N2C_CL1_loss += r3 * N2C_CL1_loss
    train_loss /= batch_num
    train_main_loss /= batch_num
    train_C2N_NL1_loss /= batch_num
    train_N2C_CL1_loss /= batch_num

    return train_loss, train_main_loss, train_C2N_NL1_loss, train_N2C_CL1_loss


#######################################################################
#                           For validation                            #
#######################################################################
def model_validate(model, validation_loader, direct, writer, dir_to_save, epoch, DEVICE):
    # initialization
    validation_loss = 0
    vali_end_loss = 0
    vali_mid_loss = 0
    validation_main_loss = 0
    validation_perceptual_loss = 0
    batch_num = 0

    avg_pesq = 0
    avg_stoi = 0

    all_batch_input = []
    all_batch_target = []
    all_batch_output = []

    # save the same sample
    clip_num = 10

    # for record the score each samples
    f_score = open(dir_to_save + '/Epoch_' + '%d_SCORES' % epoch, 'a')

    model.eval()
    with torch.no_grad():
        if cfg.latent_vector_loss:
            for inputs, targets in tools.Bar(validation_loader):
                batch_num += 1

                # to cuda
                inputs = inputs.float().to(DEVICE)
                targets = targets.float().to(DEVICE)

                # targets_mid, outputs_mid, target_spec, output_spec, outputs = model(targets, inputs, direct_mapping=direct)
                targets_mid, outputs_mid, outputs = model(targets, inputs, direct_mapping=direct)
                # targets_mid = targets_mid.permute(0, 3, 1, 2)
                # outputs_mid = outputs_mid.permute(0, 3, 1, 2)
                # target_spec = target_spec.permute(0, 2, 1)
                # output_spec = output_spec.permute(0, 2, 1)
                # for i in range(len(targets_mid[0])):
                #     fig = plt.figure()
                #     ax1 = fig.add_subplot(1, 1, 1)
                #     # figure 1
                #     ax1.plot(outputs_mid[0][i][:,0].cpu().detach().numpy(), 'r', lw=2)
                #     ax1.plot(targets_mid[0][i][:,0].cpu().detach().numpy(), 'k--')
                #
                #     # ax1.legend(['Target', 'Est.'])
                #     ax1.grid()
                #
                #     plt.title('[{} Epoch] {} frame'.format(epoch - 1, i))
                #     plt.xlabel('Frequency');
                #     plt.ylabel('dB');
                #     # plt.show()
                # for i in range(200, len(target_spec[0])):
                #     fig = plt.figure()
                #     ax1 = fig.add_subplot(1, 1, 1)
                #     # figure 1
                #     ax1.plot(target_spec[0][i].cpu().detach().numpy(), 'r', lw=2)
                #     ax1.plot(output_spec[0][i].cpu().detach().numpy(), 'k--')
                #
                #     # ax1.legend(['Target', 'Est.'])
                #     ax1.grid()
                #
                #     plt.title('[{} Epoch] {} frame'.format(epoch - 1, i))
                #     plt.xlabel('Frequency');
                #     plt.ylabel('dB');
                #     # plt.show()

                main_loss = model.loss(outputs, targets)
                mid_loss = model.loss(outputs_mid, targets_mid)

                # # min-max normalization ()
                # mid_loss -= 2.3535e-12
                # mid_loss /= (2.5166e-09 - 2.3535e-12)

                # # if you want to check the scale of the loss
                # print('main loss: {:.4} mid loss: {:.4}'.format(main_loss, mid_loss))

                r1 = 1
                r2 = 1
                r = r1 + r2
                loss = (r1 * main_loss + r2 * mid_loss) / r

                # estimate the output speech with pesq and stoi
                estimated_wavs = outputs.cpu().detach().numpy()
                clean_wavs = targets.cpu().detach().numpy()

                pesq = cal_pesq(estimated_wavs, clean_wavs)
                stoi = cal_stoi(estimated_wavs, clean_wavs)

                # pesq: 0.1 better / stoi: 0.01 better
                for i in range(len(pesq)):
                    f_score.write('PESQ {:.6f} | STOI {:.6f}\n'.format(pesq[i], stoi[i]))

                # reshape for sum
                pesq = np.reshape(pesq, (1, -1))
                stoi = np.reshape(stoi, (1, -1))

                avg_pesq += sum(pesq[0]) / len(inputs)
                avg_stoi += sum(stoi[0]) / len(inputs)

                validation_loss += loss
                vali_end_loss += r1 * main_loss
                vali_mid_loss += r2 * mid_loss

                # for saving the sample we want to tensorboard
                if epoch % 10 == 0:
                    # all batch data array
                    all_batch_input.extend(inputs)
                    all_batch_target.extend(targets)
                    all_batch_output.extend(outputs)

            # save the samples to tensorboard
            if epoch % 10 == 0:
                writer.save_samples_we_want('clip: ' + str(clip_num), all_batch_input[clip_num],
                                            all_batch_target[clip_num],
                                            all_batch_output[clip_num], epoch)

            avg_pesq /= batch_num
            avg_stoi /= batch_num

            validation_loss /= batch_num
            vali_end_loss /= batch_num
            vali_mid_loss /= batch_num

            return validation_loss, vali_end_loss, vali_mid_loss, avg_pesq, avg_stoi
        else:
            if cfg.perceptual == 'PAM':
                for inputs, targets, GMT in tools.Bar(validation_loader):
                    batch_num += 1

                    # to cuda
                    inputs = inputs.float().to(DEVICE)
                    targets = targets.float().to(DEVICE)
                    GMT = GMT.float().to(DEVICE)

                    _, _, outputs = model(inputs, direct_mapping=direct)
                    main_loss = model.loss(outputs, targets)
                    perceptual_loss = model.loss(outputs, targets, GMT=GMT, perceptual=True)

                    # the constraint ratio
                    r1 = 10e8 * 6
                    r2 = 1
                    r3 = r1 + r2
                    loss = (r1 * main_loss + r2 * perceptual_loss) / r3
                    # print('M {:.4}  P {:.4}'.format(r1 * main_loss, r2 * perceptual_loss))

                    validation_loss += loss
                    validation_main_loss += r1 * main_loss
                    validation_perceptual_loss += r2 * perceptual_loss

                    # for saving the sample we want to tensorboard
                    if epoch % 10 == 0:
                        # all batch data array
                        all_batch_input.extend(inputs)
                        all_batch_target.extend(targets)
                        all_batch_output.extend(outputs)

                    # estimate the output speech with pesq and stoi
                    estimated_wavs = outputs.cpu().detach().numpy()
                    clean_wavs = targets.cpu().detach().numpy()

                    pesq = cal_pesq(estimated_wavs, clean_wavs)
                    stoi = cal_stoi(estimated_wavs, clean_wavs)

                    # pesq: 0.1 better / stoi: 0.01 better
                    for i in range(len(pesq)):
                        f_score.write('PESQ {:.6f} | STOI {:.6f}\n'.format(pesq[i], stoi[i]))

                    # reshape for sum
                    pesq = np.reshape(pesq, (1, -1))
                    stoi = np.reshape(stoi, (1, -1))

                    avg_pesq += sum(pesq[0]) / len(inputs)
                    avg_stoi += sum(stoi[0]) / len(inputs)
                # save the samples to tensorboard
                if epoch % 10 == 0:
                    writer.save_samples_we_want('clip: ' + str(clip_num), all_batch_input[clip_num],
                                                all_batch_target[clip_num],
                                                all_batch_output[clip_num], epoch)

                avg_pesq /= batch_num
                avg_stoi /= batch_num

                validation_loss /= batch_num
                validation_main_loss /= batch_num
                validation_perceptual_loss /= batch_num

                return validation_loss, validation_main_loss, validation_perceptual_loss, avg_pesq, avg_stoi
            elif cfg.perceptual == 'LMS' or cfg.perceptual == 'PMSQE':
                for inputs, targets in tools.Bar(validation_loader):
                    batch_num += 1

                    # to cuda
                    inputs = inputs.float().to(DEVICE)
                    targets = targets.float().to(DEVICE)

                    real_spec, img_spec, outputs = model(inputs, direct_mapping=direct)
                    main_loss = model.loss(outputs, targets)
                    perceptual_loss = model.loss(outputs, targets, real_spec, img_spec, perceptual=True)

                    # the constraint ratio
                    r1 = 1
                    r2 = 1
                    r3 = r1 + r2
                    loss = (r1 * main_loss + r2 * perceptual_loss) / r3

                    # estimate the output speech with pesq and stoi
                    estimated_wavs = outputs.cpu().detach().numpy()
                    clean_wavs = targets.cpu().detach().numpy()

                    pesq = cal_pesq(estimated_wavs, clean_wavs)
                    stoi = cal_stoi(estimated_wavs, clean_wavs)

                    # pesq: 0.1 better / stoi: 0.01 better
                    for i in range(len(pesq)):
                        f_score.write('PESQ {:.6f} | STOI {:.6f}\n'.format(pesq[i], stoi[i]))

                    # reshape for sum
                    pesq = np.reshape(pesq, (1, -1))
                    stoi = np.reshape(stoi, (1, -1))

                    avg_pesq += sum(pesq[0]) / len(inputs)
                    avg_stoi += sum(stoi[0]) / len(inputs)

                    validation_loss += loss
                    validation_main_loss += main_loss
                    validation_perceptual_loss += perceptual_loss

                    # for saving the sample we want to tensorboard
                    if epoch % 10 == 0:
                        # all batch data array
                        all_batch_input.extend(inputs)
                        all_batch_target.extend(targets)
                        all_batch_output.extend(outputs)

                # save the samples to tensorboard
                if epoch % 10 == 0:
                    writer.save_samples_we_want('clip: ' + str(clip_num), all_batch_input[clip_num],
                                                all_batch_target[clip_num],
                                                all_batch_output[clip_num], epoch)
                avg_pesq /= batch_num
                avg_stoi /= batch_num

                validation_loss /= batch_num
                validation_main_loss /= batch_num
                validation_perceptual_loss /= batch_num

                return validation_loss, validation_main_loss, validation_perceptual_loss, avg_pesq, avg_stoi
            else:
                for inputs, targets in tools.Bar(validation_loader):
                    batch_num += 1

                    # to cuda
                    inputs = inputs.float().to(DEVICE)
                    targets = targets.float().to(DEVICE)

                    # if direct:
                    #     _, _, outputs, targets = model(inputs, targets, direct_mapping=direct)
                    # else:
                    #     _, _, outputs = model(inputs, direct_mapping=direct)

                    _, _, outputs = model(inputs, direct_mapping=direct)
                    loss = model.loss(outputs, targets)

                    validation_loss += loss

                    # estimate the output speech with pesq and stoi
                    estimated_wavs = outputs.cpu().detach().numpy()
                    clean_wavs = targets.cpu().detach().numpy()

                    pesq = cal_pesq(estimated_wavs, clean_wavs)
                    stoi = cal_stoi(estimated_wavs, clean_wavs)

                    # pesq: 0.1 better / stoi: 0.01 better
                    for i in range(len(pesq)):
                        f_score.write('PESQ {:.6f} | STOI {:.6f}\n'.format(pesq[i], stoi[i]))

                    # reshape for sum
                    pesq = np.reshape(pesq, (1, -1))
                    stoi = np.reshape(stoi, (1, -1))

                    avg_pesq += sum(pesq[0]) / len(inputs)
                    avg_stoi += sum(stoi[0]) / len(inputs)

                    # for saving the sample we want to tensorboard
                    if epoch % 10 == 0:
                        # all batch data array
                        all_batch_input.extend(inputs)
                        all_batch_target.extend(targets)
                        all_batch_output.extend(outputs)

                # save the samples to tensorboard
                if epoch % 10 == 0:
                    writer.save_samples_we_want('clip: ' + str(clip_num), all_batch_input[clip_num],
                                                all_batch_target[clip_num],
                                                all_batch_output[clip_num], epoch)

                validation_loss /= batch_num
                avg_pesq /= batch_num
                avg_stoi /= batch_num

                return validation_loss, avg_pesq, avg_stoi


def model_direct_validate(model, validation_loader, writer, dir_to_save, epoch, DEVICE):
    # initialization
    validation_loss = 0
    vali_end_loss = 0
    vali_mid_loss = 0
    validation_main_loss = 0
    validation_perceptual_loss = 0
    batch_num = 0

    avg_pesq = 0
    avg_stoi = 0

    all_batch_input = []
    all_batch_target = []
    all_batch_output = []

    # save the same sample
    clip_num = 10

    # for record the score each samples
    f_score = open(dir_to_save + '/Epoch_' + '%d_SCORES' % epoch, 'a')

    model.eval()
    with torch.no_grad():
        if cfg.latent_vector_loss:
            for inputs, targets in tools.Bar(validation_loader):
                batch_num += 1

                # to cuda
                inputs = inputs.float().to(DEVICE)
                targets = targets.float().to(DEVICE)

                # targets_mid, outputs_mid, target_spec, output_spec, outputs = model(targets, inputs, direct_mapping=direct)
                targets_mid, outputs_mid, outputs = model(targets, inputs, direct_mapping=direct)
                # targets_mid = targets_mid.permute(0, 3, 1, 2)
                # outputs_mid = outputs_mid.permute(0, 3, 1, 2)
                # target_spec = target_spec.permute(0, 2, 1)
                # output_spec = output_spec.permute(0, 2, 1)
                # for i in range(len(targets_mid[0])):
                #     fig = plt.figure()
                #     ax1 = fig.add_subplot(1, 1, 1)
                #     # figure 1
                #     ax1.plot(outputs_mid[0][i][:,0].cpu().detach().numpy(), 'r', lw=2)
                #     ax1.plot(targets_mid[0][i][:,0].cpu().detach().numpy(), 'k--')
                #
                #     # ax1.legend(['Target', 'Est.'])
                #     ax1.grid()
                #
                #     plt.title('[{} Epoch] {} frame'.format(epoch - 1, i))
                #     plt.xlabel('Frequency');
                #     plt.ylabel('dB');
                #     # plt.show()
                # for i in range(200, len(target_spec[0])):
                #     fig = plt.figure()
                #     ax1 = fig.add_subplot(1, 1, 1)
                #     # figure 1
                #     ax1.plot(target_spec[0][i].cpu().detach().numpy(), 'r', lw=2)
                #     ax1.plot(output_spec[0][i].cpu().detach().numpy(), 'k--')
                #
                #     # ax1.legend(['Target', 'Est.'])
                #     ax1.grid()
                #
                #     plt.title('[{} Epoch] {} frame'.format(epoch - 1, i))
                #     plt.xlabel('Frequency');
                #     plt.ylabel('dB');
                #     # plt.show()

                main_loss = model.loss(outputs, targets)
                mid_loss = model.loss(outputs_mid, targets_mid)

                # # min-max normalization ()
                # mid_loss -= 2.3535e-12
                # mid_loss /= (2.5166e-09 - 2.3535e-12)

                # # if you want to check the scale of the loss
                # print('main loss: {:.4} mid loss: {:.4}'.format(main_loss, mid_loss))

                r1 = 1
                r2 = 1
                r = r1 + r2
                loss = (r1 * main_loss + r2 * mid_loss) / r

                # estimate the output speech with pesq and stoi
                estimated_wavs = outputs.cpu().detach().numpy()
                clean_wavs = targets.cpu().detach().numpy()

                pesq = cal_pesq(estimated_wavs, clean_wavs)
                stoi = cal_stoi(estimated_wavs, clean_wavs)

                # pesq: 0.1 better / stoi: 0.01 better
                for i in range(len(pesq)):
                    f_score.write('PESQ {:.6f} | STOI {:.6f}\n'.format(pesq[i], stoi[i]))

                # reshape for sum
                pesq = np.reshape(pesq, (1, -1))
                stoi = np.reshape(stoi, (1, -1))

                avg_pesq += sum(pesq[0]) / len(inputs)
                avg_stoi += sum(stoi[0]) / len(inputs)

                validation_loss += loss
                vali_end_loss += r1 * main_loss
                vali_mid_loss += r2 * mid_loss

                # for saving the sample we want to tensorboard
                if epoch % 10 == 0:
                    # all batch data array
                    all_batch_input.extend(inputs)
                    all_batch_target.extend(targets)
                    all_batch_output.extend(outputs)

            # save the samples to tensorboard
            if epoch % 10 == 0:
                writer.save_samples_we_want('clip: ' + str(clip_num), all_batch_input[clip_num],
                                            all_batch_target[clip_num],
                                            all_batch_output[clip_num], epoch)

            avg_pesq /= batch_num
            avg_stoi /= batch_num

            validation_loss /= batch_num
            vali_end_loss /= batch_num
            vali_mid_loss /= batch_num

            return validation_loss, vali_end_loss, vali_mid_loss, avg_pesq, avg_stoi
        else:
            if cfg.perceptual == 'PAM':
                for inputs, targets, GMT in tools.Bar(validation_loader):
                    batch_num += 1

                    # to cuda
                    inputs = inputs.float().to(DEVICE)
                    targets = targets.float().to(DEVICE)
                    GMT = GMT.float().to(DEVICE)

                    _, _, outputs = model(inputs, direct_mapping=direct)
                    main_loss = model.loss(outputs, targets)
                    perceptual_loss = model.loss(outputs, targets, GMT=GMT, perceptual=True)

                    # the constraint ratio
                    r1 = 10e8 * 6
                    r2 = 1
                    r3 = r1 + r2
                    loss = (r1 * main_loss + r2 * perceptual_loss) / r3
                    # print('M {:.4}  P {:.4}'.format(r1 * main_loss, r2 * perceptual_loss))

                    validation_loss += loss
                    validation_main_loss += r1 * main_loss
                    validation_perceptual_loss += r2 * perceptual_loss

                    # for saving the sample we want to tensorboard
                    if epoch % 10 == 0:
                        # all batch data array
                        all_batch_input.extend(inputs)
                        all_batch_target.extend(targets)
                        all_batch_output.extend(outputs)

                    # estimate the output speech with pesq and stoi
                    estimated_wavs = outputs.cpu().detach().numpy()
                    clean_wavs = targets.cpu().detach().numpy()

                    pesq = cal_pesq(estimated_wavs, clean_wavs)
                    stoi = cal_stoi(estimated_wavs, clean_wavs)

                    # pesq: 0.1 better / stoi: 0.01 better
                    for i in range(len(pesq)):
                        f_score.write('PESQ {:.6f} | STOI {:.6f}\n'.format(pesq[i], stoi[i]))

                    # reshape for sum
                    pesq = np.reshape(pesq, (1, -1))
                    stoi = np.reshape(stoi, (1, -1))

                    avg_pesq += sum(pesq[0]) / len(inputs)
                    avg_stoi += sum(stoi[0]) / len(inputs)
                # save the samples to tensorboard
                if epoch % 10 == 0:
                    writer.save_samples_we_want('clip: ' + str(clip_num), all_batch_input[clip_num],
                                                all_batch_target[clip_num],
                                                all_batch_output[clip_num], epoch)

                avg_pesq /= batch_num
                avg_stoi /= batch_num

                validation_loss /= batch_num
                validation_main_loss /= batch_num
                validation_perceptual_loss /= batch_num

                return validation_loss, validation_main_loss, validation_perceptual_loss, avg_pesq, avg_stoi
            elif cfg.perceptual == 'LMS' or cfg.perceptual == 'PMSQE':
                for inputs, targets in tools.Bar(validation_loader):
                    batch_num += 1

                    # to cuda
                    inputs = inputs.float().to(DEVICE)
                    targets = targets.float().to(DEVICE)

                    real_spec, img_spec, outputs = model(inputs, direct_mapping=direct)
                    main_loss = model.loss(outputs, targets)
                    perceptual_loss = model.loss(outputs, targets, real_spec, img_spec, perceptual=True)

                    # the constraint ratio
                    r1 = 1
                    r2 = 1
                    r3 = r1 + r2
                    loss = (r1 * main_loss + r2 * perceptual_loss) / r3

                    # estimate the output speech with pesq and stoi
                    estimated_wavs = outputs.cpu().detach().numpy()
                    clean_wavs = targets.cpu().detach().numpy()

                    pesq = cal_pesq(estimated_wavs, clean_wavs)
                    stoi = cal_stoi(estimated_wavs, clean_wavs)

                    # pesq: 0.1 better / stoi: 0.01 better
                    for i in range(len(pesq)):
                        f_score.write('PESQ {:.6f} | STOI {:.6f}\n'.format(pesq[i], stoi[i]))

                    # reshape for sum
                    pesq = np.reshape(pesq, (1, -1))
                    stoi = np.reshape(stoi, (1, -1))

                    avg_pesq += sum(pesq[0]) / len(inputs)
                    avg_stoi += sum(stoi[0]) / len(inputs)

                    validation_loss += loss
                    validation_main_loss += main_loss
                    validation_perceptual_loss += perceptual_loss

                    # for saving the sample we want to tensorboard
                    if epoch % 10 == 0:
                        # all batch data array
                        all_batch_input.extend(inputs)
                        all_batch_target.extend(targets)
                        all_batch_output.extend(outputs)

                # save the samples to tensorboard
                if epoch % 10 == 0:
                    writer.save_samples_we_want('clip: ' + str(clip_num), all_batch_input[clip_num],
                                                all_batch_target[clip_num],
                                                all_batch_output[clip_num], epoch)
                avg_pesq /= batch_num
                avg_stoi /= batch_num

                validation_loss /= batch_num
                validation_main_loss /= batch_num
                validation_perceptual_loss /= batch_num

                return validation_loss, validation_main_loss, validation_perceptual_loss, avg_pesq, avg_stoi
            else:
                for inputs, targets in tools.Bar(validation_loader):
                    batch_num += 1

                    # to cuda
                    inputs = inputs.float().to(DEVICE)
                    targets = targets.float().to(DEVICE)

                    output_real, target_real, output_imag, target_imag, outputs = model(inputs, targets)
                    real_loss = model.loss(output_real, target_real)
                    imag_loss = model.loss(output_imag, target_imag)
                    loss = (real_loss + imag_loss) / 2

                    validation_loss += loss

                    # estimate the output speech with pesq and stoi
                    estimated_wavs = outputs.cpu().detach().numpy()
                    clean_wavs = targets.cpu().detach().numpy()

                    pesq = cal_pesq(estimated_wavs, clean_wavs)
                    stoi = cal_stoi(estimated_wavs, clean_wavs)

                    # pesq: 0.1 better / stoi: 0.01 better
                    for i in range(len(pesq)):
                        f_score.write('PESQ {:.6f} | STOI {:.6f}\n'.format(pesq[i], stoi[i]))

                    # reshape for sum
                    pesq = np.reshape(pesq, (1, -1))
                    stoi = np.reshape(stoi, (1, -1))

                    avg_pesq += sum(pesq[0]) / len(inputs)
                    avg_stoi += sum(stoi[0]) / len(inputs)

                    # for saving the sample we want to tensorboard
                    if epoch % 10 == 0:
                        # all batch data array
                        all_batch_input.extend(inputs)
                        all_batch_target.extend(targets)
                        all_batch_output.extend(outputs)

                # save the samples to tensorboard
                if epoch % 10 == 0:
                    writer.save_samples_we_want('clip: ' + str(clip_num), all_batch_input[clip_num],
                                                all_batch_target[clip_num],
                                                all_batch_output[clip_num], epoch)

                validation_loss /= batch_num
                avg_pesq /= batch_num
                avg_stoi /= batch_num

                return validation_loss, avg_pesq, avg_stoi


def cycle_model_validate(N2C, C2N, validation_loader, direct, writer, epoch, DEVICE):
    # initialization
    validation_loss = 0
    validation_main_loss = 0
    validation_C2N_NL1_loss = 0
    validation_N2C_CL1_loss = 0
    batch_num = 0

    all_batch_input = []
    all_batch_target = []
    all_batch_clean_output = []
    all_batch_noisy_output = []

    # save the same sample
    clip_num = 10

    N2C.eval()
    C2N.eval()
    with torch.no_grad():
        for inputs, targets in tools.Bar(validation_loader):
            batch_num += 1

            # to cuda
            inputs = inputs.float().to(DEVICE)
            targets = targets.float().to(DEVICE)

            _, _, estimated_clean_outputs = N2C(inputs, direct_mapping=direct)
            _, _, fake_noisy_outputs = C2N(estimated_clean_outputs, direct_mapping=True)

            _, _, estimated_noisy_outputs = C2N(targets, direct_mapping=True)
            _, _, fake_clean_outputs = N2C(estimated_noisy_outputs, direct_mapping=direct)

            main_loss = N2C.loss(estimated_clean_outputs, targets)

            C2N_NL1_loss = L1Loss(fake_noisy_outputs, inputs)
            N2C_CL1_loss = L1Loss(fake_clean_outputs, targets)

            # constraint ratio
            r1 = 150
            r2 = 1
            r3 = 1
            r = r1 + r2 + r3

            loss = (r1 * main_loss + r2 * C2N_NL1_loss + r3 * N2C_CL1_loss) / r

            # for saving the sample we want to tensorboard
            # if epoch % 10 == 0:
            #     # all batch data array
            #     all_batch_input.extend(inputs)
            #     all_batch_target.extend(targets)
            #     all_batch_clean_output.extend(estimated_clean_outputs)
            #     all_batch_noisy_output.extend(estimated_noisy_outputs)

            validation_loss += loss
            validation_main_loss += r1 * main_loss
            validation_C2N_NL1_loss += r2 * C2N_NL1_loss
            validation_N2C_CL1_loss += r3 * N2C_CL1_loss
        validation_loss /= batch_num
        validation_main_loss /= batch_num
        validation_C2N_NL1_loss /= batch_num
        validation_N2C_CL1_loss /= batch_num

        # save the samples to tensorboard
        # if epoch % 10 == 0:
        #     writer.save_cycle_samples_we_want('clip: ' + str(clip_num), all_batch_input[clip_num],
        #                                       all_batch_target[clip_num], all_batch_clean_output[clip_num],
        #                                       all_batch_noisy_output[clip_num], epoch)
    return validation_loss, validation_main_loss, validation_C2N_NL1_loss, validation_N2C_CL1_loss


#######################################################################
#                           For evaluation                            #
#######################################################################
def model_eval(model, validation_loader, direct, dir_to_save, epoch, DEVICE):
    # initialize
    batch_num = 0
    avg_pesq = 0
    avg_stoi = 0

    # for record the score each samples
    f_score = open(dir_to_save + '/Epoch_' + '%d_SCORES' % epoch, 'a')

    model.eval()
    with torch.no_grad():
        if cfg.perceptual == 'PAM':
            for inputs, targets, GMT in tools.Bar(validation_loader):
                batch_num += 1

                # to cuda
                inputs = inputs.float().to(DEVICE)
                targets = targets.float().to(DEVICE)

                _, _, outputs = model(inputs, direct_mapping=direct)

                if epoch % 1 == 0:
                    remain_noise = outputs - targets
                    remain_noise = remain_noise.float().to(DEVICE)
                    pam_pw_draw(inputs, targets, outputs, remain_noise, GMT, dir_to_save, cfg.fs, cfg.fft_len, epoch)

                # estimate the output speech with pesq and stoi
                estimated_wavs = outputs.cpu().detach().numpy()
                clean_wavs = targets.cpu().detach().numpy()

                pesq = cal_pesq(estimated_wavs, clean_wavs)
                stoi = cal_stoi(estimated_wavs, clean_wavs)

                # pesq: 0.1 better / stoi: 0.01 better
                for i in range(len(pesq)):
                    f_score.write('PESQ {:.6f} | STOI {:.6f}\n'.format(pesq[i], stoi[i]))

                # reshape for sum
                pesq = np.reshape(pesq, (1, -1))
                stoi = np.reshape(stoi, (1, -1))

                avg_pesq += sum(pesq[0]) / len(inputs)
                avg_stoi += sum(stoi[0]) / len(inputs)
        else:
            for inputs, targets in tools.Bar(validation_loader):
                batch_num += 1

                # to cuda
                inputs = inputs.float().to(DEVICE)
                targets = targets.float().to(DEVICE)

                if cfg.latent_vector_loss:
                    _, _, outputs = model(targets, inputs, direct_mapping=direct)
                else:
                    _, _, outputs = model(inputs, direct_mapping=direct)

                # estimate the output speech with pesq and stoi
                estimated_wavs = outputs.cpu().detach().numpy()
                clean_wavs = targets.cpu().detach().numpy()

                pesq = cal_pesq(estimated_wavs, clean_wavs)
                stoi = cal_stoi(estimated_wavs, clean_wavs)

                # pesq: 0.1 better / stoi: 0.01 better
                for i in range(len(pesq)):
                    f_score.write('PESQ {:.6f} | STOI {:.6f}\n'.format(pesq[i], stoi[i]))

                # reshape for sum
                pesq = np.reshape(pesq, (1, -1))
                stoi = np.reshape(stoi, (1, -1))

                avg_pesq += sum(pesq[0]) / len(inputs)
                avg_stoi += sum(stoi[0]) / len(inputs)
        avg_pesq /= batch_num
        avg_stoi /= batch_num
    f_score.close()
    return avg_pesq, avg_stoi
