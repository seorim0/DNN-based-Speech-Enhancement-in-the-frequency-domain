import os
import time
import torch
import shutil
import numpy as np
import config as cfg
import itertools
from model import complex_model, complex_model_latent_vector, complex_conditional_model, complex_model_direct
from write_on_tensorboard import Writer
from dataloader import create_dataloader
from trainer import model_train, model_conditional_train, model_validate, model_direct_train, model_direct_validate, \
    cycle_model_train, cycle_model_validate, model_eval


###############################################################################
#                        Helper function definition                           #
###############################################################################
# Write training related parameters into the log file.
def write_status_to_log_file(fp, total_parameters):
    fp.write('%d-%d-%d %d:%d:%d\n' %
             (time.localtime().tm_year, time.localtime().tm_mon,
              time.localtime().tm_mday, time.localtime().tm_hour,
              time.localtime().tm_min, time.localtime().tm_sec))
    fp.write('total params   : %d (%.2f M, %.2f MBytes)\n' %
             (total_parameters,
              total_parameters / 1000000.0,
              total_parameters * 4.0 / 1000000.0))


# Calculate the size of total network.
def calculate_total_params(our_model):
    total_parameters = 0
    for variable in our_model.parameters():
        shape = variable.size()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim
        total_parameters += variable_parameters

    return total_parameters


###############################################################################
#         Parameter Initialization and Setting for model training             #
###############################################################################
# Set device
DEVICE = torch.device(cfg.DEVICE)

if cfg.cycle:
    # Set model
    N2C = complex_model().to(DEVICE)
    C2N = complex_model().to(DEVICE)
    # Set optimizer and learning rate
    optimizer = torch.optim.Adam(itertools.chain(N2C.parameters(), C2N.parameters()), lr=cfg.learning_rate)
    total_params = calculate_total_params(N2C) + calculate_total_params(C2N)
elif cfg.latent_vector_loss:
    # Set model
    model = complex_model_latent_vector().to(DEVICE)
    # Set optimizer and learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    total_params = calculate_total_params(model)
elif cfg.conditional_learning:
    # Set model
    model = complex_conditional_model().to(DEVICE)
    # Set optimizer and learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    total_params = calculate_total_params(model)
elif cfg.masking_mode == 'Direct(None make)':
    direct = True
    # Set model
    model = complex_model_direct().to(DEVICE)
    # Set optimizer and learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    total_params = calculate_total_params(model)
else:
    # Set model
    model = complex_model().to(DEVICE)
    # Set optimizer and learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    total_params = calculate_total_params(model)

###############################################################################
#                        Confirm model information                            #
###############################################################################
print('%d-%d-%d %d:%d:%d\n' %
      (time.localtime().tm_year, time.localtime().tm_mon,
       time.localtime().tm_mday, time.localtime().tm_hour,
       time.localtime().tm_min, time.localtime().tm_sec))
print('total params   : %d (%.2f M, %.2f MBytes)\n' %
      (total_params,
       total_params / 1000000.0,
       total_params * 4.0 / 1000000.0))

###############################################################################
#                              Create Dataloader                              #
###############################################################################
train_loader = create_dataloader(mode='train')
validation_loader = create_dataloader(mode='valid')

###############################################################################
#                        Set a log file to store progress.                    #
#               Set a hps file to store hyper-parameters information.         #
###############################################################################
if cfg.chkpt_model is not None:  # Load the checkpoint
    print('Resuming from checkpoint: %s' % cfg.chkpt_path)

    # Set a log file to store progress.
    dir_to_save = cfg.job_dir + cfg.chkpt_model
    dir_to_logs = cfg.logs_dir + cfg.chkpt_model

    if cfg.cycle:
        N2C_checkpoint = torch.load(dir_to_save + '/N2C_chkpt_' + cfg.chkpt + '.pt')
        C2N_checkpoint = torch.load(dir_to_save + '/C2N_chkpt_' + cfg.chkpt + '.pt')
        N2C.load_state_dict(N2C_checkpoint['model'])
        C2N.load_state_dict(C2N_checkpoint['model'])
        optimizer.load_state_dict(N2C_checkpoint['optimizer'])
        epoch_start_idx = N2C_checkpoint['epoch'] + 1
    else:
        checkpoint = torch.load(cfg.chkpt_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_start_idx = checkpoint['epoch'] + 1
    mse_vali_total = np.load(str(dir_to_save + '/mse_vali_total.npy'))
    # if the loaded length is shorter than I expected, extend the length
    if len(mse_vali_total) < cfg.max_epochs:
        plus = cfg.max_epochs - len(mse_vali_total)
        mse_vali_total = np.concatenate((mse_vali_total, np.zeros(plus)), 0)
else:  # First learning
    print('Starting new training run...')
    epoch_start_idx = 1
    mse_vali_total = np.zeros(cfg.max_epochs)

    # Set a log file to store progress.
    dir_to_save = cfg.job_dir + cfg.expr_num + '_%d.%d' % (time.localtime().tm_mon,
                                                           time.localtime().tm_mday) + '_%s' % cfg.model + '_%s' % cfg.loss
    dir_to_logs = cfg.logs_dir + cfg.expr_num + '_%d.%d' % (time.localtime().tm_mon,
                                                            time.localtime().tm_mday) + '_%s' % cfg.model + '_%s' % cfg.loss

# make the file directory
if not os.path.exists(dir_to_save):
    os.mkdir(dir_to_save)
    os.mkdir(dir_to_logs)

# logging
log_fname = str(dir_to_save + '/log.txt')
if not os.path.exists(log_fname):
    fp = open(log_fname, 'w')
    write_status_to_log_file(fp, total_params)
else:
    fp = open(log_fname, 'a')

# Set a config file to store hyper-parameters information.
hps_fname = str(dir_to_save + '/config_setting.txt')
if not os.path.exists(hps_fname):
    fp_h = open(hps_fname, 'w')
else:
    fp_h = open(hps_fname, 'a')
    fp_h.write('\n\n')
with open('config.py', 'r') as f:
    hp_str = ''.join(f.readlines())
fp_h.write(hp_str)
fp_h.close()
###############################################################################
###############################################################################
#                             Main program start !!                           #
###############################################################################
###############################################################################
# Writer initialize
writer = Writer(dir_to_logs)

###############################################################################
#                                    Train                                    #
###############################################################################
if cfg.cycle:
    for epoch in range(epoch_start_idx, cfg.max_epochs):
        start_time = time.time()
        # training
        train_toral_loss, train_main_loss, train_C2N_NL1_loss, train_N2C_CL1_loss = \
            cycle_model_train(N2C, C2N, optimizer, train_loader, direct, DEVICE)

        # save checkpoint file to resume training
        save_path = str(dir_to_save + '/' + ('N2C_chkpt_%d.pt' % epoch))
        torch.save({
            'model': N2C.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }, save_path)
        save_path = str(dir_to_save + '/' + ('C2N_chkpt_%d.pt' % epoch))
        torch.save({
            'model': C2N.state_dict(),
            'epoch': epoch
        }, save_path)

        # validation
        vali_toral_loss, vali_main_loss, vali_C2N_NL1_loss, vali_N2C_CL1_loss = \
            cycle_model_validate(N2C, C2N, validation_loader, direct, writer, epoch, DEVICE)

        mse_vali_total[epoch - 1] = vali_toral_loss
        np.save(str(dir_to_save + '/mse_vali_total.npy'), mse_vali_total)

        # write the loss on tensorboard
        writer.log_loss(train_toral_loss, vali_toral_loss, epoch)
        writer.log_main_loss(train_main_loss, vali_main_loss, epoch)
        writer.log_C2N_loss(train_C2N_NL1_loss, vali_C2N_NL1_loss, epoch)
        writer.log_N2C_loss(train_N2C_CL1_loss, vali_N2C_CL1_loss, epoch)

        print('Epoch [{}] | T {:.6f} | V {:.6} takes {:.2f} seconds'
              .format(epoch, train_toral_loss, vali_toral_loss, time.time() - start_time))
        print('          | main loss: T {:.6f} V {:.6f} | C2N loss: T {:.6f} V {:.6f} | C2N loss: T {:.6f} V {:.6f}'
              .format(train_main_loss, vali_main_loss,
                      train_C2N_NL1_loss, vali_C2N_NL1_loss, train_N2C_CL1_loss, vali_N2C_CL1_loss))
        # log file save
        fp.write('Epoch [{}] | T {:.6f} | V {:.6} takes {:.2f} seconds\n'
                 .format(epoch, train_toral_loss, vali_toral_loss, time.time() - start_time))
        fp.write(
            '          | main loss: T {:.6f} V {:.6f} | C2N loss: T {:.6f} V {:.6f} | C2N loss: T {:.6f} V {:.6f}\n'
            .format(train_main_loss, vali_main_loss,
                    train_C2N_NL1_loss, vali_C2N_NL1_loss, train_N2C_CL1_loss, vali_N2C_CL1_loss))

        # It takes quite some time to evaluate the scores for each epoch, so you can control the frequency
        if epoch % 10 == 0:
            vali_pesq, vali_stoi = model_eval(N2C, validation_loader, direct, dir_to_save, epoch, DEVICE)
            # write the loss on tensorboard per 5 epochs
            writer.log_score(vali_pesq, vali_stoi, epoch)
            print('           | PESQ: V {:.6f} | STOI: V {:.6f} '.format(vali_pesq, vali_stoi))
            fp.write('           | PESQ: V {:.6f} | STOI: V {:.6f} \n'.format(vali_pesq, vali_stoi))
elif cfg.latent_vector_loss:
    for epoch in range(epoch_start_idx, cfg.max_epochs):
        start_time = time.time()
        # Training
        train_loss, train_end_loss, train_mid_loss = model_train(model, optimizer, train_loader, direct, DEVICE)

        # save checkpoint file to resume training
        save_path = str(dir_to_save + '/' + ('chkpt_%d.pt' % epoch))
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }, save_path)

        # Validation
        vali_loss, vali_end_loss, vali_mid_loss, vali_pesq, vali_stoi \
            = model_validate(model, validation_loader, direct, writer, dir_to_save, epoch, DEVICE)
        # write the loss on tensorboard
        writer.log_trial_loss(train_loss, train_end_loss, train_mid_loss, vali_loss, vali_end_loss, vali_mid_loss, epoch)
        writer.log_score(vali_pesq, vali_stoi, epoch)

        print('Epoch [{}] | T {:.6f} {:.6f} {:.6f} | V {:.6} {:.6f} {:.6f} takes {:.2f} seconds\n'
              .format(epoch, train_loss, train_end_loss, train_mid_loss, vali_loss, vali_end_loss,
                      vali_mid_loss, time.time() - start_time))
        print('          | V PESQ: {:.6f} | STOI: {:.6f} '.format(vali_pesq, vali_stoi))
        # log file save
        fp.write('Epoch [{}] | T {:.6f} {:.6f} {:.6f} | V {:.6} {:.6f} {:.6f} takes {:.2f} seconds\n'
                 .format(epoch, train_loss, train_end_loss, train_mid_loss, vali_loss, vali_end_loss,
                         vali_mid_loss, time.time() - start_time))

        fp.write('          | V PESQ: {:.6f} | STOI: {:.6f} \n'.format(vali_pesq, vali_stoi))

        mse_vali_total[epoch - 1] = vali_loss
        np.save(str(dir_to_save + '/mse_vali_total.npy'), mse_vali_total)

        # # It takes quite some time to evaluate the scores for each epoch, so you can control the frequency
        # if epoch % 1 == 0:
        #     vali_pesq, vali_stoi = model_eval(model, validation_loader, direct, dir_to_save, epoch, DEVICE)
        #     # write the loss on tensorboard per 5 epochs
        #     writer.log_score(vali_pesq, vali_stoi, epoch)
        #     print('          | V PESQ: {:.6f} | STOI: {:.6f} '.format(vali_pesq, vali_stoi))
        #     fp.write('          | V PESQ: {:.6f} | STOI: {:.6f} \n'.format(vali_pesq, vali_stoi))
        
elif cfg.conditional_learning:
    for epoch in range(epoch_start_idx, cfg.max_epochs):
        start_time = time.time()
        # Training
        if cfg.perceptual != 'False':  # The case of using perceptual loss function
            train_loss, train_main_loss, train_perceptual_loss = \
                model_train(model, optimizer, train_loader, direct, DEVICE)
        else:
            train_loss, train_main_loss, train_sub_loss = \
                model_conditional_train(model, optimizer, train_loader, direct, DEVICE)

        # save checkpoint file to resume training
        save_path = str(dir_to_save + '/' + ('chkpt_%d.pt' % epoch))
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }, save_path)

        # Validation
        if cfg.perceptual != 'False':  # The case of using perceptual loss function
            vali_loss, vali_main_loss, vali_perceptual_loss, vali_pesq, vali_stoi = \
                model_validate(model, validation_loader, direct, writer, dir_to_save, epoch, DEVICE)
            # write the loss on tensorboard
            writer.log_loss(train_loss, vali_loss, epoch)
            writer.log_perceptual_loss(train_main_loss, train_perceptual_loss,
                                       vali_main_loss, vali_perceptual_loss, epoch)
            writer.log_score(vali_pesq, vali_stoi, epoch)

            print('Epoch [{}] | T {:.6f} | V {:.6} takes {:.2f} seconds\n'
                  .format(epoch, train_loss, vali_loss, time.time() - start_time))
            print('             main loss: T {:.6f} V {:.6f} | perceptual loss: T {:.6f} V {:.6f}'
                  .format(train_main_loss, vali_main_loss, train_perceptual_loss, vali_perceptual_loss))
            print('          | V PESQ: {:.6f} | STOI: {:.6f} '.format(vali_pesq, vali_stoi))
            # log file save
            fp.write('Epoch [{}] | T {:.6f} | V {:.6} takes {:.2f} seconds\n'
                     .format(epoch, train_loss, vali_loss, time.time() - start_time))
            fp.write('             main loss: T {:.6f} V {:.6f} | perceptual loss: T {:.6f} V {:.6f}'
                     .format(train_main_loss, vali_main_loss, train_perceptual_loss, vali_perceptual_loss))
            fp.write('          | V PESQ: {:.6f} | STOI: {:.6f} \n'.format(vali_pesq, vali_stoi))
        else:
            # vali_loss, vali_main_loss, vali_sub_loss, vali_pesq, vali_stoi = \
            #     model_conditional_validate(model, validation_loader, direct, writer, dir_to_save, epoch, DEVICE)
            vali_loss, vali_pesq, vali_stoi = \
                model_conditional_validate(model, validation_loader, direct, writer, dir_to_save, epoch, DEVICE)
            # write the loss on tensorboard
            # writer.log_conditional_loss(train_loss, vali_loss, train_main_loss, vali_main_loss, train_sub_loss, vali_sub_loss, epoch)
            writer.log_conditional_loss(train_loss, vali_loss, train_main_loss, train_sub_loss, epoch)
            writer.log_score(vali_pesq, vali_stoi, epoch)

            print('Epoch [{}] | T {:.6f} | V {:.6} takes {:.2f} seconds\n'
                  .format(epoch, train_loss, vali_loss, time.time() - start_time))
            # print('             main loss: T {:.6f} V {:.6f} | sub loss: T {:.6f} V {:.6f}'
            #       .format(train_main_loss, vali_main_loss, train_sub_loss, vali_sub_loss))
            # print('             main loss: T {:.6f} V {:.6f} | sub loss: T {:.6f}'
            #       .format(train_main_loss, vali_main_loss, train_sub_loss))
            print('             main loss: T {:.6f} | sub loss: T {:.6f}'
                  .format(train_main_loss, train_sub_loss))
            print('          | V PESQ: {:.6f} | STOI: {:.6f} '.format(vali_pesq, vali_stoi))
            # log file save
            fp.write('Epoch [{}] | T {:.6f} | V {:.6} takes {:.2f} seconds\n'
                     .format(epoch, train_loss, vali_loss, time.time() - start_time))
            # fp.write('             main loss: T {:.6f} V {:.6f} | sub loss: T {:.6f} V {:.6f}'
            #          .format(train_main_loss, vali_main_loss, train_sub_loss, vali_sub_loss))
            # fp.write('             main loss: T {:.6f} V {:.6f} | sub loss: T {:.6f}'
            #          .format(train_main_loss, vali_main_loss, train_sub_loss))
            fp.write('             main loss: T {:.6f} | sub loss: T {:.6f}'
                     .format(train_main_loss, train_sub_loss))
            fp.write('          | V PESQ: {:.6f} | STOI: {:.6f} \n'.format(vali_pesq, vali_stoi))

        mse_vali_total[epoch - 1] = vali_loss
        np.save(str(dir_to_save + '/mse_vali_total.npy'), mse_vali_total)

        # # It takes quite some time to evaluate the scores for each epoch, so you can control the frequency
        # if epoch % 1 == 0:
        #     vali_pesq, vali_stoi = model_eval(model, validation_loader, direct, dir_to_save, epoch, DEVICE)
        #     # write the loss on tensorboard per 5 epochs
        #     writer.log_score(vali_pesq, vali_stoi, epoch)
        #     print('          | V PESQ: {:.6f} | STOI: {:.6f} '.format(vali_pesq, vali_stoi))
        #     fp.write('          | V PESQ: {:.6f} | STOI: {:.6f} \n'.format(vali_pesq, vali_stoi))
        
elif direct:
    for epoch in range(epoch_start_idx, cfg.max_epochs):
        start_time = time.time()
        # Training
        if cfg.perceptual != 'False':  # The case of using perceptual loss function
            train_loss, train_main_loss, train_perceptual_loss = \
                model_direct_train(model, optimizer, train_loader, DEVICE)
        else:
            train_loss = model_direct_train(model, optimizer, train_loader, DEVICE)

        # save checkpoint file to resume training
        save_path = str(dir_to_save + '/' + ('chkpt_%d.pt' % epoch))
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }, save_path)

        # Validation
        if cfg.perceptual != 'False':  # The case of using perceptual loss function
            vali_loss, vali_main_loss, vali_perceptual_loss, vali_pesq, vali_stoi = \
                model_direct_validate(model, validation_loader, writer, dir_to_save, epoch, DEVICE)
            # write the loss on tensorboard
            writer.log_loss(train_loss, vali_loss, epoch)
            writer.log_perceptual_loss(train_main_loss, train_perceptual_loss,
                                       vali_main_loss, vali_perceptual_loss, epoch)
            writer.log_score(vali_pesq, vali_stoi, epoch)

            print('Epoch [{}] | T {:.6f} | V {:.6} takes {:.2f} seconds\n'
                  .format(epoch, train_loss, vali_loss, time.time() - start_time))
            print('             main loss: T {:.6f} V {:.6f} | perceptual loss: T {:.6f} V {:.6f}'
                  .format(train_main_loss, vali_main_loss, train_perceptual_loss, vali_perceptual_loss))
            print('          | V PESQ: {:.6f} | STOI: {:.6f} '.format(vali_pesq, vali_stoi))
            # log file save
            fp.write('Epoch [{}] | T {:.6f} | V {:.6} takes {:.2f} seconds\n'
                     .format(epoch, train_loss, vali_loss, time.time() - start_time))
            fp.write('             main loss: T {:.6f} V {:.6f} | perceptual loss: T {:.6f} V {:.6f}'
                     .format(train_main_loss, vali_main_loss, train_perceptual_loss, vali_perceptual_loss))
            fp.write('          | V PESQ: {:.6f} | STOI: {:.6f} \n'.format(vali_pesq, vali_stoi))
        else:
            vali_loss, vali_pesq, vali_stoi = \
                model_direct_validate(model, validation_loader, writer, dir_to_save, epoch, DEVICE)
            # write the loss on tensorboard
            writer.log_loss(train_loss, vali_loss, epoch)
            writer.log_score(vali_pesq, vali_stoi, epoch)

            print('Epoch [{}] | T {:.6f} | V {:.6} takes {:.2f} seconds\n'
                  .format(epoch, train_loss, vali_loss, time.time() - start_time))
            print('          | V PESQ: {:.6f} | STOI: {:.6f} '.format(vali_pesq, vali_stoi))
            # log file save
            fp.write('Epoch [{}] | T {:.6f} | V {:.6} takes {:.2f} seconds\n'
                     .format(epoch, train_loss, vali_loss, time.time() - start_time))
            fp.write('          | V PESQ: {:.6f} | STOI: {:.6f} \n'.format(vali_pesq, vali_stoi))

        mse_vali_total[epoch - 1] = vali_loss
        np.save(str(dir_to_save + '/mse_vali_total.npy'), mse_vali_total)

        # # It takes quite some time to evaluate the scores for each epoch, so you can control the frequency
        # if epoch % 1 == 0:
        #     vali_pesq, vali_stoi = model_eval(model, validation_loader, direct, dir_to_save, epoch, DEVICE)
        #     # write the loss on tensorboard per 5 epochs
        #     writer.log_score(vali_pesq, vali_stoi, epoch)
        #     print('          | V PESQ: {:.6f} | STOI: {:.6f} '.format(vali_pesq, vali_stoi))
        #     fp.write('          | V PESQ: {:.6f} | STOI: {:.6f} \n'.format(vali_pesq, vali_stoi))
else:
    for epoch in range(epoch_start_idx, cfg.max_epochs):
        start_time = time.time()
        # Training
        if cfg.perceptual != 'False':  # The case of using perceptual loss function
            train_loss, train_main_loss, train_perceptual_loss = \
                model_train(model, optimizer, train_loader, direct, DEVICE)
        else:
            train_loss = model_train(model, optimizer, train_loader, direct, DEVICE)

        # save checkpoint file to resume training
        save_path = str(dir_to_save + '/' + ('chkpt_%d.pt' % epoch))
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }, save_path)

        # Validation
        if cfg.perceptual != 'False':  # The case of using perceptual loss function
            vali_loss, vali_main_loss, vali_perceptual_loss, vali_pesq, vali_stoi = \
                model_validate(model, validation_loader, direct, writer, dir_to_save, epoch, DEVICE)
            # write the loss on tensorboard
            writer.log_loss(train_loss, vali_loss, epoch)
            writer.log_perceptual_loss(train_main_loss, train_perceptual_loss,
                                       vali_main_loss, vali_perceptual_loss, epoch)
            writer.log_score(vali_pesq, vali_stoi, epoch)

            print('Epoch [{}] | T {:.6f} | V {:.6} takes {:.2f} seconds\n'
                  .format(epoch, train_loss, vali_loss, time.time() - start_time))
            print('             main loss: T {:.6f} V {:.6f} | perceptual loss: T {:.6f} V {:.6f}'
                  .format(train_main_loss, vali_main_loss, train_perceptual_loss, vali_perceptual_loss))
            print('          | V PESQ: {:.6f} | STOI: {:.6f} '.format(vali_pesq, vali_stoi))
            # log file save
            fp.write('Epoch [{}] | T {:.6f} | V {:.6} takes {:.2f} seconds\n'
                     .format(epoch, train_loss, vali_loss, time.time() - start_time))
            fp.write('             main loss: T {:.6f} V {:.6f} | perceptual loss: T {:.6f} V {:.6f}'
                     .format(train_main_loss, vali_main_loss, train_perceptual_loss, vali_perceptual_loss))
            fp.write('          | V PESQ: {:.6f} | STOI: {:.6f} \n'.format(vali_pesq, vali_stoi))
        else:
            vali_loss, vali_pesq, vali_stoi = \
                model_validate(model, validation_loader, direct, writer, dir_to_save, epoch, DEVICE)
            # write the loss on tensorboard
            writer.log_loss(train_loss, vali_loss, epoch)
            writer.log_score(vali_pesq, vali_stoi, epoch)

            print('Epoch [{}] | T {:.6f} | V {:.6} takes {:.2f} seconds\n'
                  .format(epoch, train_loss, vali_loss, time.time() - start_time))
            print('          | V PESQ: {:.6f} | STOI: {:.6f} '.format(vali_pesq, vali_stoi))
            # log file save
            fp.write('Epoch [{}] | T {:.6f} | V {:.6} takes {:.2f} seconds\n'
                     .format(epoch, train_loss, vali_loss, time.time() - start_time))
            fp.write('          | V PESQ: {:.6f} | STOI: {:.6f} \n'.format(vali_pesq, vali_stoi))

        mse_vali_total[epoch - 1] = vali_loss
        np.save(str(dir_to_save + '/mse_vali_total.npy'), mse_vali_total)

        # # It takes quite some time to evaluate the scores for each epoch, so you can control the frequency
        # if epoch % 1 == 0:
        #     vali_pesq, vali_stoi = model_eval(model, validation_loader, direct, dir_to_save, epoch, DEVICE)
        #     # write the loss on tensorboard per 5 epochs
        #     writer.log_score(vali_pesq, vali_stoi, epoch)
        #     print('          | V PESQ: {:.6f} | STOI: {:.6f} '.format(vali_pesq, vali_stoi))
        #     fp.write('          | V PESQ: {:.6f} | STOI: {:.6f} \n'.format(vali_pesq, vali_stoi))

fp.close()
print('Training has been finished.')

# Copy optimum model that has minimum MSE.
print('Save optimum models...')
min_index = np.argmin(mse_vali_total)
print('Minimum validation loss is at ' + str(min_index + 1) + '.')
src_file = str(dir_to_save + '/' + ('chkpt_%d.pt' % (min_index + 1)))
tgt_file = str(dir_to_save + '/chkpt_opt.pt')
shutil.copy(src_file, tgt_file)
