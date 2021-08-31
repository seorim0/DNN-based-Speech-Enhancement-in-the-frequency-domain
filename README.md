# Speech enhancement with Pytorch
You can do DNN-based speech enhancement(SE) using various method through this repository.   
First, you can make noisy data by mixing clean speech and noise, and then you can generate a dataset with those data. (If you already have a dataset, skip this.) The dataset is used for deep learning training, and the type and configuration of the model can be adjusted in various ways, if you want. And, the result of the network can be evaluated through various objective metrics (PESQ, STOI, CSIG, CBAK, COVL).
   
## Networks   
> You can find a list that you can adjust in various ways at config.py, and they are:   
* Real network
** convolutional recurrent network (CRN)   
* Complex network 
** deep complex convolutional recurrent network (DCCRN) [[1]](https://arxiv.org/abs/2008.00264)  

## Learning methods
* T-F masking
* Spectral mapping

## Loss functions   
* MSE   
* SDR   
* SI-SNR   
* SI-SDR   

> and you can join the loss functions with perceptual loss.   
* LMS
* PMSQE
   
## Requirements
> This repository is tested on Ubuntu 20.04.
* Python 3.7+
* Cuda 11.1+
* CuDNN 8+
* Pytorch 1.9+
<br>

> Library
* tqdm
* asteroid   
* scipy   
* matplotlib   
* tensorboardX   
   
## Reference   
https://github.com/huyanxin/DeepComplexCRN   
https://github.com/usimarit/semetrics     
https://ecs.utdallas.edu/loizou/speech/software.htm
