"""
Where the model is actually trained and validated
"""

import torch
import numpy as np
import tools_for_model as tools
from tools_for_estimate import cal_pesq, cal_stoi


#######################################################################
#                             For train                               #
#######################################################################
# T-F masking
def model_train(model, optimizer, train_loader, DEVICE):
    # initialization
    train_loss = 0
    batch_num = 0

    # arr = []
    # train
    model.train()
    for inputs, targets in tools.Bar(train_loader):
        batch_num += 1

        # to cuda
        inputs = inputs.float().to(DEVICE)
        targets = targets.float().to(DEVICE)

        _, _, outputs = model(inputs, targets)
        loss = model.loss(outputs, targets)
        # # if you want to check the scale of the loss
        # print('loss: {:.4}'.format(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss
    train_loss /= batch_num

    return train_loss 


def model_perceptual_train(model, optimizer, train_loader, DEVICE):
    # initialization
    train_loss = 0
    train_main_loss = 0
    train_perceptual_loss = 0
    batch_num = 0

    # train
    model.train()
    for inputs, targets in tools.Bar(train_loader):
        batch_num += 1

        # to cuda
        inputs = inputs.float().to(DEVICE)
        targets = targets.float().to(DEVICE)

        real_spec, img_spec, outputs = model(inputs)
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


def fullsubnet_train(model, optimizer, train_loader, DEVICE):
    # initialization
    train_loss = 0
    batch_num = 0

    # arr = []
    # train
    model.train()
    for inputs, targets in tools.Bar(train_loader):
        batch_num += 1

        # to cuda
        inputs = inputs.float().to(DEVICE)
        targets = targets.float().to(DEVICE)

        noisy_complex = tools.stft(inputs)
        clean_complex = tools.stft(targets)

        noisy_mag, _ = tools.mag_phase(noisy_complex)
        cIRM = tools.build_complex_ideal_ratio_mask(noisy_complex, clean_complex)

        cRM = model(noisy_mag)
        loss = model.loss(cRM, cIRM)
        # # if you want to check the scale of the loss
        # print('loss: {:.4}'.format(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss
    train_loss /= batch_num

    return train_loss


# Spectral mapping
def dccrn_direct_train(model, optimizer, train_loader, DEVICE):
    # initialization
    train_loss = 0
    batch_num = 0

    # train
    model.train()
    for inputs, targets in tools.Bar(train_loader):
        batch_num += 1

        # to cuda
        inputs = inputs.float().to(DEVICE)
        targets = targets.float().to(DEVICE)

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

    return train_loss


def crn_direct_train(model, optimizer, train_loader, DEVICE):
    # initialization
    train_loss = 0
    batch_num = 0

    # train
    model.train()
    for inputs, targets in tools.Bar(train_loader):
        batch_num += 1

        # to cuda
        inputs = inputs.float().to(DEVICE)
        targets = targets.float().to(DEVICE)

        output_mag, target_mag, _ = model(inputs, targets)
        loss = model.loss(output_mag, target_mag)

        # # if you want to check the scale of the loss
        # print('loss: {:.4}'.format(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss
    train_loss /= batch_num

    return train_loss


#######################################################################
#                           For validation                            #
#######################################################################
# T-F masking
def model_validate(model, validation_loader, writer, dir_to_save, epoch, DEVICE):
    # initialization
    validation_loss = 0
    batch_num = 0

    avg_pesq = 0
    avg_stoi = 0

    # for record the score each samples
    f_score = open(dir_to_save + '/Epoch_' + '%d_SCORES' % epoch, 'a')

    model.eval()
    with torch.no_grad():
        for inputs, targets in tools.Bar(validation_loader):
            batch_num += 1

            # to cuda
            inputs = inputs.float().to(DEVICE)
            targets = targets.float().to(DEVICE)

            _, _, outputs = model(inputs, targets)
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

        # save the samples to tensorboard
        if epoch % 10 == 0:
            writer.log_wav(inputs[0], targets[0], outputs[0], epoch)

        validation_loss /= batch_num
        avg_pesq /= batch_num
        avg_stoi /= batch_num

        return validation_loss, avg_pesq, avg_stoi
    
    
def model_perceptual_validate(model, validation_loader, writer, dir_to_save, epoch, DEVICE):
    # initialization
    validation_loss = 0
    validation_main_loss = 0
    validation_perceptual_loss = 0
    batch_num = 0

    avg_pesq = 0
    avg_stoi = 0

    # for record the score each samples
    f_score = open(dir_to_save + '/Epoch_' + '%d_SCORES' % epoch, 'a')

    model.eval()
    with torch.no_grad():
        for inputs, targets in tools.Bar(validation_loader):
            batch_num += 1

            # to cuda
            inputs = inputs.float().to(DEVICE)
            targets = targets.float().to(DEVICE)

            real_spec, img_spec, outputs = model(inputs)
            main_loss = model.loss(outputs, targets)
            perceptual_loss = model.loss(outputs, targets, real_spec, img_spec, perceptual=True)

            # the constraint ratio
            r1 = 1
            r2 = 1
            r3 = r1 + r2
            loss = (r1 * main_loss + r2 * perceptual_loss) / r3

            validation_loss += loss
            validation_main_loss += r1 * main_loss
            validation_perceptual_loss += r2 * perceptual_loss

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
            writer.log_wav(inputs[0], targets[0], outputs[0], epoch)

        validation_loss /= batch_num
        validation_main_loss /= batch_num
        validation_perceptual_loss /= batch_num
        avg_pesq /= batch_num
        avg_stoi /= batch_num

        return validation_loss, validation_main_loss, validation_perceptual_loss, avg_pesq, avg_stoi


def fullsubnet_validate(model, validation_loader, writer, dir_to_save, epoch, DEVICE):
    # initialization
    validation_loss = 0
    batch_num = 0

    avg_pesq = 0
    avg_stoi = 0

    # for record the score each samples
    f_score = open(dir_to_save + '/Epoch_' + '%d_SCORES' % epoch, 'a')

    model.eval()
    with torch.no_grad():
        for inputs, targets in tools.Bar(validation_loader):
            batch_num += 1

            # to cuda
            inputs = inputs.float().to(DEVICE)
            targets = targets.float().to(DEVICE)

            noisy_complex = tools.stft(inputs)
            clean_complex = tools.stft(targets)

            noisy_mag, _ = tools.mag_phase(noisy_complex)
            cIRM = tools.build_complex_ideal_ratio_mask(noisy_complex, clean_complex)

            cRM = model(noisy_mag)
            loss = model.loss(cRM, cIRM)

            validation_loss += loss

            # estimate the output speech with pesq and stoi
            cRM = tools.decompress_cIRM(cRM)
            enhanced_real = cRM[..., 0] * noisy_complex.real - cRM[..., 1] * noisy_complex.imag
            enhanced_imag = cRM[..., 1] * noisy_complex.real + cRM[..., 0] * noisy_complex.imag
            enhanced_complex = torch.stack((enhanced_real, enhanced_imag), dim=-1)
            enhanced_outputs = tools.istft(enhanced_complex, length=inputs.size(-1))

            estimated_wavs = enhanced_outputs.cpu().detach().numpy()
            clean_wavs = targets.cpu().detach().numpy()

            pesq = tools.cal_pesq(estimated_wavs, clean_wavs)
            stoi = tools.cal_stoi(estimated_wavs, clean_wavs)

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
            writer.log_wav(inputs[0], targets[0], enhanced_outputs[0], epoch)

        validation_loss /= batch_num
        avg_pesq /= batch_num
        avg_stoi /= batch_num

        return validation_loss, avg_pesq, avg_stoi
    
    
# Spectral mapping
def dccrn_direct_validate(model, validation_loader, writer, dir_to_save, epoch, DEVICE):
    # initialization
    validation_loss = 0
    batch_num = 0

    avg_pesq = 0
    avg_stoi = 0

    # for record the score each samples
    f_score = open(dir_to_save + '/Epoch_' + '%d_SCORES' % epoch, 'a')

    model.eval()
    with torch.no_grad():
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

        # save the samples to tensorboard
        if epoch % 10 == 0:
            writer.log_wav(inputs[0], targets[0], outputs[0], epoch)
                
        validation_loss /= batch_num
        avg_pesq /= batch_num
        avg_stoi /= batch_num

        return validation_loss, avg_pesq, avg_stoi
    

def crn_direct_validate(model, validation_loader, writer, dir_to_save, epoch, DEVICE):
    # initialization
    validation_loss = 0
    batch_num = 0

    avg_pesq = 0
    avg_stoi = 0

    # for record the score each samples
    f_score = open(dir_to_save + '/Epoch_' + '%d_SCORES' % epoch, 'a')

    model.eval()
    with torch.no_grad():
        for inputs, targets in tools.Bar(validation_loader):
            batch_num += 1

            # to cuda
            inputs = inputs.float().to(DEVICE)
            targets = targets.float().to(DEVICE)

            output_mag, target_mag, outputs = model(inputs, targets)
            loss = model.loss(output_mag, target_mag)

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

        # save the samples to tensorboard
        if epoch % 10 == 0:
            writer.log_wav(inputs[0], targets[0], outputs[0], epoch)
                
        validation_loss /= batch_num
        avg_pesq /= batch_num
        avg_stoi /= batch_num

        return validation_loss, avg_pesq, avg_stoi
