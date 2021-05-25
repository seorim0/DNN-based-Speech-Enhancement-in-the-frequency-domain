"""
for data preprocessing
"""

# setup
fs = 16000
snr_set = [0, 5, 10, 15, 20]

data_name = ""
mode = "train"  # train | validation | test

noise_subset = ['chime2_lounge', 'chime3_bus', 'chime3_cafe', 'chime3_pedestrain', 'chime3_street',
                          'noisex92_babble', 'noisex92_factory', 'noisex92_military',
                          'noisex92_pink', 'noisex92_volvo', 'noisex92_white']
# ['etsi_inside_aircraft', 'etsi_kindergarten', 'etsi_mensa', 'etsi_schoolyard',
#  'etsi_shopping_center', 'etsi_train_station', 'tank']

# for test setup
padding = True
test_noise_type = "seen"  # seen | unseen


