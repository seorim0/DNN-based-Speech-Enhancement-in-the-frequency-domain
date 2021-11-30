"""
Configuration for train_interface

You can check the essential information,
and if you want to change model structure or training method,
you have to change this file.
"""
#######################################################################
#                                 path                                #
#######################################################################
job_dir = './models/'
logs_dir = './logs/'
chkpt_model = None  # 'FILE PATH (if you have pretrained model..)'
chkpt = str("EPOCH")
if chkpt_model is not None:
    chkpt_path = job_dir + chkpt_model + '/chkpt_' + chkpt + '.pt'

#######################################################################
#                         possible setting                            #
#######################################################################
# the list you can do
model_list = ['DCCRN', 'CRN', 'FullSubNet']
loss_list = ['MSE', 'SDR', 'SI-SNR', 'SI-SDR']
perceptual_list = [False, 'LMS', 'PMSQE']
lstm_type = ['real', 'complex']
main_net = ['LSTM', 'GRU']
mask_type = ['Direct(None make)', 'E', 'C', 'R']

# experiment number setting
expr_num = 'EXPERIMENT_NUMBER'
DEVICE = 'cuda'  # if you want to run the code with 'cpu', change 'cpu'
#######################################################################
#                           current setting                           #
#######################################################################
model = model_list[0]
loss = loss_list[1]
perceptual = perceptual_list[0]
lstm = lstm_type[1]
sequence_model = main_net[0]

masking_mode = mask_type[1]
skip_type = True   # False, if you want to remove 'skip connection'

# hyper-parameters
max_epochs = 100
learning_rate = 0.001
batch = 10

# kernel size
dccrn_kernel_num = [32, 64, 128, 256, 256, 256]
#######################################################################
#                         model information                           #
#######################################################################
fs = 16000
win_len = 400
win_inc = 100
ola_ratio = 0.75
fft_len = 512
sam_sec = fft_len / fs
frm_samp = fs * (fft_len / fs)
window = 'hanning'

# for DCCRN
rnn_layers = 2
rnn_units = 256

# for CRN
rnn_input_size = 512

# for FullSubNet
sb_num_neighbors = 15
fb_num_neighbors = 0
num_freqs = fft_len // 2 + 1
look_ahead = 2
fb_output_activate_function = "ReLU"
sb_output_activate_function = None
fb_model_hidden_size = 512
sb_model_hidden_size = 384
weight_init = False
norm_type = "offline_laplace_norm"
num_groups_in_drop_band = 2
#######################################################################
#                      setting error check                            #
#######################################################################
# if the setting is wrong, print error message
assert not (masking_mode == 'Direct(None make)' and perceptual is not False), \
    "This setting is not created "
assert not (model == 'FullSubNet' and perceptual is not False), \
    "This setting is not created "

#######################################################################
#                           print setting                             #
#######################################################################
print('--------------------  C  O  N  F  I  G  ----------------------')
print('--------------------------------------------------------------')
print('MODEL INFO : {}'.format(model))
print('LOSS INFO : {}, perceptual : {}'.format(loss, perceptual))
if model != 'FullSubNet':
    print('LSTM : {}'.format(lstm))
    print('SKIP : {}'.format(skip_type))
    print('MASKING INFO : {}'.format(masking_mode))
else:
    print('Main network : {}'.format(sequence_model))
print('\nBATCH : {}'.format(batch))
print('LEARNING RATE : {}'.format(learning_rate))
print('--------------------------------------------------------------')
print('--------------------------------------------------------------\n')
