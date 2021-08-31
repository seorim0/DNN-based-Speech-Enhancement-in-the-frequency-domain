# Speech Enhancement with Pytorch
You can do DNN-based speech enhancement(SE) in the frequency domain using various method through this repository.   
First, you have to make noisy data by mixing clean speech and noise. The dataset is used for deep learning training.   
And, you can adjust the type of the network and configuration in various ways, as shown below.   
The results of the network can be evaluated through various objective metrics (PESQ, STOI, CSIG, CBAK, COVL).



<!-- You can change -->
<details open="open">
  <summary>You can change</summary>
  <ol>
    <li>
      <a href="#networks">Networks</a>
    </li>
    <li>
      <a href="#learning-methods">Learning methods</a>
    </li>
    <li><a href="#loss-functions">Loss functions</a></li>
  </ol>
</details>
<br>


## Requirements
> This repository is tested on Ubuntu 20.04, and
* Python 3.7
* Cuda 11.1
* CuDNN 8.0.5
* Pytorch 1.9.0


## Getting Started   

## Tutorials
'SE_tutorials.ipynb' was made for tutorial.   
You can simply train the CRN with the colab file.   


<!-- NETWORKS -->
## Networks   
> You can find a list that you can adjust in various ways at config.py, and they are:   
* Real network   
   - convolutional recurrent network (CRN)   
* Complex network   
   - deep complex convolutional recurrent network (DCCRN) [[1]](https://arxiv.org/abs/2008.00264)  

<!-- LEARNING METHODS -->
## Learning Methods
* T-F masking
* Spectral mapping

<!-- LOSS FUNCTIONS -->
## Loss Functions   
* MSE   
* SDR   
* SI-SNR   
* SI-SDR   

> and you can join the loss functions with perceptual loss.   
* LMS
* PMSQE


## Tensorboard
> As shown below, you can check whether the network is being trained well in real time through 'write_on_tensorboard.py'.   

![tensor](https://user-images.githubusercontent.com/55497506/131444707-4459a979-8652-46f4-82f1-0c640cfff685.png)   
* loss
* pesq, stoi
* spectrogram
  

## Reference   
https://github.com/huyanxin/DeepComplexCRN   
https://github.com/usimarit/semetrics     
https://ecs.utdallas.edu/loizou/speech/software.htm
