# Speech enhancement with complex network
You can do DNN-based SE using a complex network through this repository. First, you can make noisy data by mixing clean speech and noise, and then you can generate a dataset with those data. (If you already have a dataset, skip this.) The dataset is used for deep learning training, and the type and configuration of the model can be adjusted in various ways, if you want. And, the result of the network can be evaluated through various objective metrics (PESQ, STOI, CSIG, CBAK, COVL).

You can find a list that you can adjust in various ways at config.py, and they are:
Models (Networks)
- cCRN
- DCUNET
- DCCRN
Loss functions
- MSE
- SDR
- SI-SNR
- SI-SDR
you can join the loss functions with perceptual loss.
- LMS
- PMSQE

# Readme to be completed soon..


- Reference   
https://github.com/huyanxin/DeepComplexCRN   
https://github.com/usimarit/semetrics     
https://ecs.utdallas.edu/loizou/speech/software.htm
