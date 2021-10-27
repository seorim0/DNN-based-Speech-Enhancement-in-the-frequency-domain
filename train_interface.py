import os
import time
import torch
import shutil
import numpy as np
import config as cfg
from models import DCCRN, CRN, FullSubNet  # you can import 'DCCRN' or 'CRN' or 'FullSubNet'
from write_on_tensorboard import Writer
from dataloader import create_dataloader
from trainer import model_train, model_validate, \
    model_perceptual_train, model_perceptual_validate, \
    dccrn_direct_train, dccrn_direct_validate, \
    crn_direct_train, crn_direct_validate, \
    fullsubnet_train, fullsubnet_validate
    

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

# Set model
if cfg.model == 'DCCRN':
    model = DCCRN().to(DEVICE)
elif cfg.model == 'CRN':
    model = CRN().to(DEVICE)
elif cfg.model == 'FullSubNet':
    model = FullSubNet().to(DEVICE)
# Set optimizer and learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
total_params = calculate_total_params(model)

# Set trainer and estimator
if cfg.perceptual is not False:
    trainer = model_perceptual_train
    estimator = model_perceptual_validate
elif cfg.model == 'FullSubNet':
    trainer = fullsubnet_train
    estimator = fullsubnet_validate
elif cfg.masking_mode == 'Direct(None make)' and cfg.model == 'DCCRN':
    trainer = dccrn_direct_train
    estimator = dccrn_direct_validate
elif cfg.masking_mode == 'Direct(None make)' and cfg.model == 'CRN':
    trainer = crn_direct_train
    estimator = crn_direct_validate
else:
    trainer = model_train
    estimator = model_validate

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
    
    # make the file directory to save the models
    if not os.path.exists(cfg.job_dir):
        os.mkdir(cfg.job_dir)
    if not os.path.exists(cfg.logs_dir):
        os.mkdir(cfg.logs_dir)
        
    epoch_start_idx = 1
    mse_vali_total = np.zeros(cfg.max_epochs)

    # Set a log file to store progress.
    dir_to_save = cfg.job_dir + cfg.expr_num + '_%d.%d' % (time.localtime().tm_mon, time.localtime().tm_mday) + \
                  '_%s' % cfg.model + '_%s' % cfg.loss
    dir_to_logs = cfg.logs_dir + cfg.expr_num + '_%d.%d' % (time.localtime().tm_mon, time.localtime().tm_mday) \
                  + '_%s' % cfg.model + '_%s' % cfg.loss

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
if cfg.perceptual is not False:  # train with perceptual loss function
    for epoch in range(epoch_start_idx, cfg.max_epochs):
        start_time = time.time()
        # Training
        train_loss, train_main_loss, train_perceptual_loss = trainer(model, optimizer, train_loader, DEVICE)

        # save checkpoint file to resume training
        save_path = str(dir_to_save + '/' + ('chkpt_%d.pt' % epoch))
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }, save_path)

        # Validation
        vali_loss, validation_main_loss, validation_perceptual_loss, vali_pesq, vali_stoi = \
            estimator(model, validation_loader, writer, dir_to_save, epoch, DEVICE)
        # write the loss on tensorboard
        writer.log_loss(train_loss, vali_loss, epoch)
        writer.log_score(vali_pesq, vali_stoi, epoch)
        writer.log_sub_loss(train_main_loss, train_perceptual_loss,
                            validation_main_loss, validation_perceptual_loss, epoch)

        print('Epoch [{}] | T {:.6f} | V {:.6} '
              .format(epoch, train_loss, vali_loss))
        print('          | T {:.6f} {:.6f} | V {:.6} {:.6f} takes {:.2f} seconds\n'
              .format(epoch, train_main_loss, train_perceptual_loss, validation_main_loss, validation_perceptual_loss,
                      time.time() - start_time))
        print('          | V PESQ: {:.6f} | STOI: {:.6f} '.format(vali_pesq, vali_stoi))
        # log file save
        fp.write('Epoch [{}] | T {:.6f} | V {:.6}\n'
                 .format(epoch, train_loss, vali_loss))
        fp.write('          | T {:.6f} {:.6f} | V {:.6} {:.6f} takes {:.2f} seconds\n'
                 .format(epoch, train_main_loss, train_perceptual_loss,
                         validation_main_loss, validation_perceptual_loss, time.time() - start_time))
        fp.write('          | V PESQ: {:.6f} | STOI: {:.6f} \n'.format(vali_pesq, vali_stoi))

        mse_vali_total[epoch - 1] = vali_loss
        np.save(str(dir_to_save + '/mse_vali_total.npy'), mse_vali_total)
else:
    for epoch in range(epoch_start_idx, cfg.max_epochs):
        start_time = time.time()
        # Training
        train_loss = trainer(model, optimizer, train_loader, DEVICE)

        # save checkpoint file to resume training
        save_path = str(dir_to_save + '/' + ('chkpt_%d.pt' % epoch))
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }, save_path)

        # Validation
        vali_loss, vali_pesq, vali_stoi = \
            estimator(model, validation_loader, writer, dir_to_save, epoch, DEVICE)
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

fp.close()
print('Training has been finished.')

# Copy optimum model that has minimum MSE.
print('Save optimum models...')
min_index = np.argmin(mse_vali_total)
print('Minimum validation loss is at ' + str(min_index + 1) + '.')
src_file = str(dir_to_save + '/' + ('chkpt_%d.pt' % (min_index + 1)))
tgt_file = str(dir_to_save + '/chkpt_opt.pt')
shutil.copy(src_file, tgt_file)
