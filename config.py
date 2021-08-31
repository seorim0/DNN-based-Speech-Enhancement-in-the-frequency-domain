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
model_list = ['DCCRN', 'CRN']
loss_list = ['MSE', 'SDR', 'SI-SNR', 'SI-SDR']
perceptual_list = [False, 'LMS', 'PMSQE']
lstm_type = ['real', 'complex']
mask_type = ['Direct(None make)', 'E', 'C', 'R']
window_type = ['hanning']

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

masking_mode = mask_type[1]
window = window_type[0]
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
ola_ratio = 1 - win_inc / win_len
fft_len = 512
sam_sec = fft_len / fs
frm_samp = fs * (fft_len / fs)

rnn_layers = 2
rnn_units = 256

# for CRN
rnn_input_size = 640
#######################################################################
#                      setting error check                            #
#######################################################################
# if the setting is wrong, print error message
assert not (masking_mode == 'Direct(None make)' and perceptual is not False), \
    "This setting is not created "

#######################################################################
#                           print setting                             #
#######################################################################
print('--------------------  C  O  N  F  I  G  ----------------------')
print('--------------------------------------------------------------')
print('MODEL INFO : {}'.format(model))
print('LOSS INFO : {}, perceptual : {}'.format(loss, perceptual))
print('LSTM : {}'.format(lstm))
print('SKIP : {}'.format(skip_type))
print('MASKING INFO : {}'.format(masking_mode))
print('\nBATCH : {}'.format(batch))
print('LEARNING RATE : {}'.format(learning_rate))
print('--------------------------------------------------------------')
print('--------------------------------------------------------------\n')
