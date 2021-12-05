import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F
import torch.nn.init as init
from scipy.signal import get_window
import matplotlib.pylab as plt
import config as cfg


############################################################################
#                         for convolutional STFT                           #
############################################################################
# this is from conv_stft https://github.com/huyanxin/DeepComplexCRN
def init_kernels(win_len, win_inc, fft_len, win_type=None, invers=False):
    if win_type == 'None' or win_type is None:
        window = np.ones(win_len)
    else:
        window = get_window(win_type, win_len, fftbins=True)  # **0.5

    N = fft_len
    fourier_basis = np.fft.rfft(np.eye(N))[:win_len]
    real_kernel = np.real(fourier_basis)
    imag_kernel = np.imag(fourier_basis)
    kernel = np.concatenate([real_kernel, imag_kernel], 1).T

    if invers:
        kernel = np.linalg.pinv(kernel).T

    kernel = kernel * window
    kernel = kernel[:, None, :]
    return torch.from_numpy(kernel.astype(np.float32)), torch.from_numpy(window[None, :, None].astype(np.float32))


class ConvSTFT(nn.Module):

    def __init__(self, win_len, win_inc, fft_len=None, win_type='hamming', feature_type='real', fix=True):
        super(ConvSTFT, self).__init__()

        if fft_len == None:
            self.fft_len = np.int(2 ** np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len

        kernel, _ = init_kernels(win_len, win_inc, self.fft_len, win_type)
        # self.weight = nn.Parameter(kernel, requires_grad=(not fix))
        self.register_buffer('weight', kernel)
        self.feature_type = feature_type
        self.stride = win_inc
        self.win_len = win_len
        self.dim = self.fft_len

    def forward(self, inputs):
        if inputs.dim() == 2:
            inputs = torch.unsqueeze(inputs, 1)
        inputs = F.pad(inputs, [self.win_len - self.stride, self.win_len - self.stride])
        outputs = F.conv1d(inputs, self.weight, stride=self.stride)

        if self.feature_type == 'complex':
            return outputs
        else:
            dim = self.dim // 2 + 1
            real = outputs[:, :dim, :]
            imag = outputs[:, dim:, :]
            mags = torch.sqrt(real ** 2 + imag ** 2)
            phase = torch.atan2(imag, real)
            return mags, phase


class ConviSTFT(nn.Module):

    def __init__(self, win_len, win_inc, fft_len=None, win_type='hamming', feature_type='real', fix=True):
        super(ConviSTFT, self).__init__()
        if fft_len == None:
            self.fft_len = np.int(2 ** np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len
        kernel, window = init_kernels(win_len, win_inc, self.fft_len, win_type, invers=True)
        # self.weight = nn.Parameter(kernel, requires_grad=(not fix))
        self.register_buffer('weight', kernel)
        self.feature_type = feature_type
        self.win_type = win_type
        self.win_len = win_len
        self.stride = win_inc
        self.dim = self.fft_len
        self.register_buffer('window', window)
        self.register_buffer('enframe', torch.eye(win_len)[:, None, :])

    def forward(self, inputs, phase=None):
        """
        inputs : [B, N+2, T] (complex spec) or [B, N//2+1, T] (mags)
        phase: [B, N//2+1, T] (if not none)
        """

        if phase is not None:
            real = inputs * torch.cos(phase)
            imag = inputs * torch.sin(phase)
            inputs = torch.cat([real, imag], 1)

        outputs = F.conv_transpose1d(inputs, self.weight, stride=self.stride)

        # this is from torch-stft: https://github.com/pseeth/torch-stft
        t = self.window.repeat(1, 1, inputs.size(-1)) ** 2
        coff = F.conv_transpose1d(t, self.enframe, stride=self.stride)

        outputs = outputs / (coff + 1e-8)

        # # outputs = torch.where(coff == 0, outputs, outputs/coff)
        outputs = outputs[..., self.win_len - self.stride:-(self.win_len - self.stride)]

        return outputs


############################################################################
#                             for complex rnn                              #
############################################################################
def get_casual_padding1d():
    pass


def get_casual_padding2d():
    pass


class cPReLU(nn.Module):

    def __init__(self, complex_axis=1):
        super(cPReLU, self).__init__()
        self.r_prelu = nn.PReLU()
        self.i_prelu = nn.PReLU()
        self.complex_axis = complex_axis

    def forward(self, inputs):
        real, imag = torch.chunk(inputs, 2, self.complex_axis)
        real = self.r_prelu(real)
        imag = self.i_prelu(imag)
        return torch.cat([real, imag], self.complex_axis)


class NavieComplexLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, projection_dim=None, bidirectional=False, batch_first=False):
        super(NavieComplexLSTM, self).__init__()

        self.input_dim = input_size // 2
        self.rnn_units = hidden_size // 2
        self.real_lstm = nn.LSTM(self.input_dim, self.rnn_units, num_layers=1, bidirectional=bidirectional,
                                 batch_first=False)
        self.imag_lstm = nn.LSTM(self.input_dim, self.rnn_units, num_layers=1, bidirectional=bidirectional,
                                 batch_first=False)
        if bidirectional:
            bidirectional = 2
        else:
            bidirectional = 1
        if projection_dim is not None:
            self.projection_dim = projection_dim // 2
            self.r_trans = nn.Linear(self.rnn_units * bidirectional, self.projection_dim)
            self.i_trans = nn.Linear(self.rnn_units * bidirectional, self.projection_dim)
        else:
            self.projection_dim = None

    def forward(self, inputs):
        if isinstance(inputs, list):
            real, imag = inputs
        elif isinstance(inputs, torch.Tensor):
            real, imag = torch.chunk(inputs, -1)
        r2r_out = self.real_lstm(real)[0]
        r2i_out = self.imag_lstm(real)[0]
        i2r_out = self.real_lstm(imag)[0]
        i2i_out = self.imag_lstm(imag)[0]
        real_out = r2r_out - i2i_out
        imag_out = i2r_out + r2i_out
        if self.projection_dim is not None:
            real_out = self.r_trans(real_out)
            imag_out = self.i_trans(imag_out)
        # print(real_out.shape,imag_out.shape)
        return [real_out, imag_out]

    def flatten_parameters(self):
        self.imag_lstm.flatten_parameters()
        self.real_lstm.flatten_parameters()


def complex_cat(inputs, axis):
    real, imag = [], []
    for idx, data in enumerate(inputs):
        r, i = torch.chunk(data, 2, axis)
        real.append(r)
        imag.append(i)
    real = torch.cat(real, axis)
    imag = torch.cat(imag, axis)
    outputs = torch.cat([real, imag], axis)
    return outputs


############################################################################
#                         for convolutional layer                          #
############################################################################
class ComplexConv2d(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            dilation=1,
            groups=1,
            causal=True,
            complex_axis=1,
    ):
        '''
            in_channels: real+imag
            out_channels: real+imag
            kernel_size : input [B,C,D,T] kernel size in [D,T]
            padding : input [B,C,D,T] padding in [D,T]
            causal: if causal, will padding time dimension's left side,
                    otherwise both

        '''
        super(ComplexConv2d, self).__init__()
        self.in_channels = in_channels // 2
        self.out_channels = out_channels // 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.causal = causal
        self.groups = groups
        self.dilation = dilation
        self.complex_axis = complex_axis

        self.real_conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size, self.stride,
                                   padding=[self.padding[0], 0], dilation=self.dilation, groups=self.groups)
        self.imag_conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size, self.stride,
                                   padding=[self.padding[0], 0], dilation=self.dilation, groups=self.groups)

        nn.init.normal_(self.real_conv.weight.data, std=0.05)
        nn.init.normal_(self.imag_conv.weight.data, std=0.05)
        nn.init.constant_(self.real_conv.bias, 0.)
        nn.init.constant_(self.imag_conv.bias, 0.)

    def forward(self, inputs):
        if self.padding[1] != 0 and self.causal:
            inputs = F.pad(inputs, [self.padding[1], 0, 0, 0])  # # [width left, width right, height left, height right]
        else:
            inputs = F.pad(inputs, [self.padding[1], self.padding[1], 0, 0])

        if self.complex_axis == 0:
            real = self.real_conv(inputs)
            imag = self.imag_conv(inputs)
            real2real, imag2real = torch.chunk(real, 2, self.complex_axis)
            real2imag, imag2imag = torch.chunk(imag, 2, self.complex_axis)

        else:
            if isinstance(inputs, torch.Tensor):
                real, imag = torch.chunk(inputs, 2, self.complex_axis)

            real2real = self.real_conv(real, )
            imag2imag = self.imag_conv(imag, )

            real2imag = self.imag_conv(real)
            imag2real = self.real_conv(imag)

        real = real2real - imag2imag
        imag = real2imag + imag2real
        out = torch.cat([real, imag], self.complex_axis)

        return out


class ComplexConvTranspose2d(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            output_padding=(0, 0),
            causal=False,
            complex_axis=1,
            groups=1
    ):
        '''
            in_channels: real+imag
            out_channels: real+imag
        '''
        super(ComplexConvTranspose2d, self).__init__()
        self.in_channels = in_channels // 2
        self.out_channels = out_channels // 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        self.real_conv = nn.ConvTranspose2d(self.in_channels, self.out_channels, kernel_size, self.stride,
                                            padding=self.padding, output_padding=output_padding, groups=self.groups)
        self.imag_conv = nn.ConvTranspose2d(self.in_channels, self.out_channels, kernel_size, self.stride,
                                            padding=self.padding, output_padding=output_padding, groups=self.groups)

        self.complex_axis = complex_axis

        nn.init.normal_(self.real_conv.weight.data, std=0.05)
        nn.init.normal_(self.imag_conv.weight.data, std=0.05)
        nn.init.constant_(self.real_conv.bias, 0.)
        nn.init.constant_(self.imag_conv.bias, 0.)

    def forward(self, inputs):

        if isinstance(inputs, torch.Tensor):
            real, imag = torch.chunk(inputs, 2, self.complex_axis)
        elif isinstance(inputs, tuple) or isinstance(inputs, list):
            real = inputs[0]
            imag = inputs[1]
        if self.complex_axis == 0:
            real = self.real_conv(inputs)
            imag = self.imag_conv(inputs)
            real2real, imag2real = torch.chunk(real, 2, self.complex_axis)
            real2imag, imag2imag = torch.chunk(imag, 2, self.complex_axis)

        else:
            if isinstance(inputs, torch.Tensor):
                real, imag = torch.chunk(inputs, 2, self.complex_axis)

            real2real = self.real_conv(real, )
            imag2imag = self.imag_conv(imag, )

            real2imag = self.imag_conv(real)
            imag2real = self.real_conv(imag)

        real = real2real - imag2imag
        imag = real2imag + imag2real
        out = torch.cat([real, imag], self.complex_axis)

        return out


class RealConv2d(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            dilation=1,
            groups=1,
            causal=True,
            complex_axis=1,
    ):
        '''
            in_channels: real+imag
            out_channels: real+imag
            kernel_size : input [B,C,D,T] kernel size in [D,T]
            padding : input [B,C,D,T] padding in [D,T]
            causal: if causal, will padding time dimension's left side,
                    otherwise both

        '''
        super(RealConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.causal = causal
        self.groups = groups
        self.dilation = dilation

        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size, self.stride,
                              padding=[self.padding[0], 0], dilation=self.dilation, groups=self.groups)

        nn.init.normal_(self.conv.weight.data, std=0.05)
        nn.init.constant_(self.conv.bias, 0.)

    def forward(self, inputs):
        if self.padding[1] != 0 and self.causal:
            inputs = F.pad(inputs, [self.padding[1], 0, 0, 0])  ## [width left, width right, height left, height right]
        else:
            inputs = F.pad(inputs, [self.padding[1], self.padding[1], 0, 0])

        out = self.conv(inputs)

        return out


class RealConvTranspose2d(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            output_padding=(0, 0),
            groups=1
    ):
        '''
            in_channels: real+imag
            out_channels: real+imag
        '''
        super(RealConvTranspose2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        self.conv = nn.ConvTranspose2d(self.in_channels, self.out_channels, kernel_size, self.stride,
                                       padding=self.padding, output_padding=output_padding, groups=self.groups)

        nn.init.normal_(self.conv.weight.data, std=0.05)
        nn.init.constant_(self.conv.bias, 0.)

    def forward(self, inputs):
        out = self.conv(inputs)

        return out


# Source: https://github.com/ChihebTrabelsi/deep_complex_networks/tree/pytorch
# from https://github.com/IMLHF/SE_DCUNet/blob/f28bf1661121c8901ad38149ea827693f1830715/models/layers/complexnn.py#L55
class ComplexBatchNorm(torch.nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, complex_axis=1):
        super(ComplexBatchNorm, self).__init__()
        self.num_features = num_features // 2
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.complex_axis = complex_axis

        if self.affine:
            self.Wrr = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Wri = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Wii = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Br = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Bi = torch.nn.Parameter(torch.Tensor(self.num_features))
        else:
            self.register_parameter('Wrr', None)
            self.register_parameter('Wri', None)
            self.register_parameter('Wii', None)
            self.register_parameter('Br', None)
            self.register_parameter('Bi', None)

        if self.track_running_stats:
            self.register_buffer('RMr', torch.zeros(self.num_features))
            self.register_buffer('RMi', torch.zeros(self.num_features))
            self.register_buffer('RVrr', torch.ones(self.num_features))
            self.register_buffer('RVri', torch.zeros(self.num_features))
            self.register_buffer('RVii', torch.ones(self.num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('RMr', None)
            self.register_parameter('RMi', None)
            self.register_parameter('RVrr', None)
            self.register_parameter('RVri', None)
            self.register_parameter('RVii', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.RMr.zero_()
            self.RMi.zero_()
            self.RVrr.fill_(1)
            self.RVri.zero_()
            self.RVii.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.Br.data.zero_()
            self.Bi.data.zero_()
            self.Wrr.data.fill_(1)
            self.Wri.data.uniform_(-.9, +.9)  # W will be positive-definite
            self.Wii.data.fill_(1)

    def _check_input_dim(self, xr, xi):
        assert (xr.shape == xi.shape)
        assert (xr.size(1) == self.num_features)

    def forward(self, inputs):
        # self._check_input_dim(xr, xi)

        xr, xi = torch.chunk(inputs, 2, axis=self.complex_axis)
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        #
        # NOTE: The precise meaning of the "training flag" is:
        #       True:  Normalize using batch   statistics, update running statistics
        #              if they are being collected.
        #       False: Normalize using running statistics, ignore batch   statistics.
        #
        training = self.training or not self.track_running_stats
        redux = [i for i in reversed(range(xr.dim())) if i != 1]
        vdim = [1] * xr.dim()
        vdim[1] = xr.size(1)

        #
        # Mean M Computation and Centering
        #
        # Includes running mean update if training and running.
        #
        if training:
            Mr, Mi = xr, xi
            for d in redux:
                Mr = Mr.mean(d, keepdim=True)
                Mi = Mi.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RMr.lerp_(Mr.squeeze(), exponential_average_factor)
                self.RMi.lerp_(Mi.squeeze(), exponential_average_factor)
        else:
            Mr = self.RMr.view(vdim)
            Mi = self.RMi.view(vdim)
        xr, xi = xr - Mr, xi - Mi

        #
        # Variance Matrix V Computation
        #
        # Includes epsilon numerical stabilizer/Tikhonov regularizer.
        # Includes running variance update if training and running.
        #
        if training:
            Vrr = xr * xr
            Vri = xr * xi
            Vii = xi * xi
            for d in redux:
                Vrr = Vrr.mean(d, keepdim=True)
                Vri = Vri.mean(d, keepdim=True)
                Vii = Vii.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RVrr.lerp_(Vrr.squeeze(), exponential_average_factor)
                self.RVri.lerp_(Vri.squeeze(), exponential_average_factor)
                self.RVii.lerp_(Vii.squeeze(), exponential_average_factor)
        else:
            Vrr = self.RVrr.view(vdim)
            Vri = self.RVri.view(vdim)
            Vii = self.RVii.view(vdim)
        Vrr = Vrr + self.eps
        Vri = Vri
        Vii = Vii + self.eps

        #
        # Matrix Inverse Square Root U = V^-0.5
        #
        # sqrt of a 2x2 matrix,
        # - https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
        tau = Vrr + Vii
        delta = torch.addcmul(Vrr * Vii, -1, Vri, Vri)
        s = delta.sqrt()
        t = (tau + 2 * s).sqrt()

        # matrix inverse, http://mathworld.wolfram.com/MatrixInverse.html
        rst = (s * t).reciprocal()
        Urr = (s + Vii) * rst
        Uii = (s + Vrr) * rst
        Uri = (- Vri) * rst

        #
        # Optionally left-multiply U by affine weights W to produce combined
        # weights Z, left-multiply the inputs by Z, then optionally bias them.
        #
        # y = Zx + B
        # y = WUx + B
        # y = [Wrr Wri][Urr Uri] [xr] + [Br]
        #     [Wir Wii][Uir Uii] [xi]   [Bi]
        #
        if self.affine:
            Wrr, Wri, Wii = self.Wrr.view(vdim), self.Wri.view(vdim), self.Wii.view(vdim)
            Zrr = (Wrr * Urr) + (Wri * Uri)
            Zri = (Wrr * Uri) + (Wri * Uii)
            Zir = (Wri * Urr) + (Wii * Uri)
            Zii = (Wri * Uri) + (Wii * Uii)
        else:
            Zrr, Zri, Zir, Zii = Urr, Uri, Uri, Uii

        yr = (Zrr * xr) + (Zri * xi)
        yi = (Zir * xr) + (Zii * xi)

        if self.affine:
            yr = yr + self.Br.view(vdim)
            yi = yi + self.Bi.view(vdim)

        outputs = torch.cat([yr, yi], self.complex_axis)
        return outputs

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)


def complex_cat(inputs, axis):
    real, imag = [], []
    for idx, data in enumerate(inputs):
        r, i = torch.chunk(data, 2, axis)
        real.append(r)
        imag.append(i)
    real = torch.cat(real, axis)
    imag = torch.cat(imag, axis)
    outputs = torch.cat([real, imag], axis)
    return outputs

############################################################################
#                           for FullSubNet                                 #
############################################################################
# Source: https://github.com/haoxiangsnr/FullSubNet
# from https://github.com/haoxiangsnr/FullSubNet/blob/main/audio_zen/model/module/sequence_model.py
# from https://github.com/haoxiangsnr/FullSubNet/blob/main/audio_zen/model/base_model.py
# from https://github.com/haoxiangsnr/FullSubNet/blob/main/audio_zen/acoustics/feature.py
def stft(y, n_fft=cfg.fft_len, hop_length=int(cfg.win_len*cfg.ola_ratio), win_length=cfg.win_len):
    """
    Args:
        y: [B, F, T]
        n_fft: num of FFT
        hop_length: hop length
        win_length: window length

    Returns:
        [B, F, T], **complex-valued** STFT coefficients

    """
    assert y.dim() == 2
    return torch.stft(
        y,
        n_fft,
        hop_length,
        win_length,
        window=torch.hann_window(win_length).to(y.device),
        return_complex=True
    )


def istft(features, n_fft=cfg.fft_len, hop_length=int(cfg.win_len*cfg.ola_ratio), win_length=cfg.win_len, length=None, use_mag_phase=False):
    """
    Wrapper for the official torch.istft

    Args:
        features: [B, F, T, 2] (complex) or ([B, F, T], [B, F, T]) (mag and phase)
        n_fft:
        hop_length:
        win_length:
        device:
        length:
        use_mag_phase: use mag and phase as inputs of iSTFT

    Returns:
        [B, T]
    """
    if use_mag_phase:
        # (mag, phase) or [mag, phase]
        assert isinstance(features, tuple) or isinstance(features, list)
        mag, phase = features
        features = torch.stack([mag * torch.cos(phase), mag * torch.sin(phase)], dim=-1)

    return torch.istft(
        features,
        n_fft,
        hop_length,
        win_length,
        window=torch.hann_window(win_length).to(features.device),
        length=length
    )


def mag_phase(complex_tensor):
    return torch.abs(complex_tensor), torch.angle(complex_tensor)


def build_complex_ideal_ratio_mask(noisy: torch.complex64, clean: torch.complex64) -> torch.Tensor:
    """

    Args:
        noisy: [B, F, T], noisy complex-valued stft coefficients
        clean: [B, F, T], clean complex-valued stft coefficients

    Returns:
        [B, F, T, 2]
    """
    denominator = torch.square(noisy.real) + torch.square(noisy.imag) + EPSILON

    mask_real = (noisy.real * clean.real + noisy.imag * clean.imag) / denominator
    mask_imag = (noisy.real * clean.imag - noisy.imag * clean.real) / denominator

    complex_ratio_mask = torch.stack((mask_real, mask_imag), dim=-1)

    return compress_cIRM(complex_ratio_mask, K=10, C=0.1)


def compress_cIRM(mask, K=10, C=0.1):
    """
        Compress from (-inf, +inf) to [-K ~ K]
    """
    if torch.is_tensor(mask):
        mask = -100 * (mask <= -100) + mask * (mask > -100)
        mask = K * (1 - torch.exp(-C * mask)) / (1 + torch.exp(-C * mask))
    else:
        mask = -100 * (mask <= -100) + mask * (mask > -100)
        mask = K * (1 - np.exp(-C * mask)) / (1 + np.exp(-C * mask))
    return mask


def decompress_cIRM(mask, K=10, limit=9.9):
    mask = limit * (mask >= limit) - limit * (mask <= -limit) + mask * (torch.abs(mask) < limit)
    mask = -K * torch.log((K - mask) / (K + mask))
    return mask


class SequenceModel(nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            hidden_size,
            num_layers,
            bidirectional,
            sequence_model="GRU",
            output_activate_function="Tanh"
    ):
        super().__init__()
        # Sequence layer
        if sequence_model == "LSTM":
            self.sequence_model = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=0.8,
            )
        elif sequence_model == "GRU":
            self.sequence_model = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=0.8,
            )
        else:
            raise NotImplementedError(f"Not implemented {sequence_model}")

        # Fully connected layer
        if bidirectional:
            self.fc_output_layer = nn.Linear(hidden_size * 2, output_size)
        else:
            self.fc_output_layer = nn.Linear(hidden_size, output_size)

        # Activation function layer
        if output_activate_function:
            if output_activate_function == "Tanh":
                self.activate_function = nn.Tanh()
            elif output_activate_function == "ReLU":
                self.activate_function = nn.ReLU()
            elif output_activate_function == "ReLU6":
                self.activate_function = nn.ReLU6()
            else:
                raise NotImplementedError(f"Not implemented activation function {self.activate_function}")

        self.output_activate_function = output_activate_function

    def forward(self, x):
        """
        Args:
            x: [B, F, T]
        Returns:
            [B, F, T]
        """
        assert x.dim() == 3
        self.sequence_model.flatten_parameters()

        x = x.permute(0, 2, 1).contiguous()  # [B, F, T] => [B, T, F]
        o, _ = self.sequence_model(x)
        o = self.fc_output_layer(o)
        if self.output_activate_function:
            o = self.activate_function(o)
        o = o.permute(0, 2, 1).contiguous()  # [B, T, F] => [B, F, T]
        return o
    
    
EPSILON = np.finfo(np.float32).eps


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    @staticmethod
    def unfold(input, num_neighbor):
        """
        Along with the frequency dim, split overlapped sub band units from spectrogram.

        Args:
            input: [B, C, F, T]
            num_neighbor:

        Returns:
            [B, N, C, F_s, T], F, e.g. [2, 161, 1, 19, 200]
        """
        assert input.dim() == 4, f"The dim of input is {input.dim()}. It should be four dim."
        batch_size, num_channels, num_freqs, num_frames = input.size()

        if num_neighbor < 1:
            # No change for the input
            return input.permute(0, 2, 1, 3).reshape(batch_size, num_freqs, num_channels, 1, num_frames)

        output = input.reshape(batch_size * num_channels, 1, num_freqs, num_frames)
        sub_band_unit_size = num_neighbor * 2 + 1

        # Pad to the top and bottom
        output = F.pad(output, [0, 0, num_neighbor, num_neighbor], mode="reflect")

        output = F.unfold(output, (sub_band_unit_size, num_frames))
        assert output.shape[-1] == num_freqs, f"n_freqs != N (sub_band), {num_freqs} != {output.shape[-1]}"

        # Split the dim of the unfolded feature
        output = output.reshape(batch_size, num_channels, sub_band_unit_size, num_frames, num_freqs)
        output = output.permute(0, 4, 1, 2, 3).contiguous()

        return output

    @staticmethod
    def _reduce_complexity_separately(sub_band_input, full_band_output, device):
        """

        Args:
            sub_band_input: [60, 257, 1, 33, 200]
            full_band_output: [60, 257, 1, 3, 200]
            device:

        Notes:
            1. 255 and 256 freq not able to be trained
            2. batch size 

        Returns:
            [60, 85, 1, 36, 200]
        """
        batch_size = full_band_output.shape[0]
        n_freqs = full_band_output.shape[1]
        sub_batch_size = batch_size // 3
        final_selected = []

        for idx in range(3):
            # [0, 60) => [0, 20)
            sub_batch_indices = torch.arange(idx * sub_batch_size, (idx + 1) * sub_batch_size, device=device)
            full_band_output_sub_batch = torch.index_select(full_band_output, dim=0, index=sub_batch_indices)
            sub_band_output_sub_batch = torch.index_select(sub_band_input, dim=0, index=sub_batch_indices)

            # Avoid to use padded value (first freq and last freq)
            # i = 0, (1, 256, 3) = [1, 4, ..., 253]
            # i = 1, (2, 256, 3) = [2, 5, ..., 254]
            # i = 2, (3, 256, 3) = [3, 6, ..., 255]
            freq_indices = torch.arange(idx + 1, n_freqs - 1, step=3, device=device)
            full_band_output_sub_batch = torch.index_select(full_band_output_sub_batch, dim=1, index=freq_indices)
            sub_band_output_sub_batch = torch.index_select(sub_band_output_sub_batch, dim=1, index=freq_indices)

            # ([30, 85, 1, 33 200], [30, 85, 1, 3, 200]) => [30, 85, 1, 36, 200]

            final_selected.append(torch.cat([sub_band_output_sub_batch, full_band_output_sub_batch], dim=-2))

        return torch.cat(final_selected, dim=0)

    @staticmethod
    def sband_forgetting_norm(input, train_sample_length):
        """
        Args:
            input:
            train_sample_length:

        Returns:

        """
        assert input.ndim == 3
        batch_size, n_freqs, n_frames = input.size()

        eps = 1e-10
        alpha = (train_sample_length - 1) / (train_sample_length + 1)
        mu = 0
        mu_list = []

        for idx in range(input.shape[-1]):
            if idx < train_sample_length:
                alp = torch.min(torch.tensor([(idx - 1) / (idx + 1), alpha]))
                mu = alp * mu + (1 - alp) * torch.mean(input[:, :, idx], dim=1).reshape(batch_size, 1)  # [B, 1]
            else:
                mu = alpha * mu + (1 - alpha) * input[:, (n_freqs // 2 - 1), idx].reshape(batch_size, 1)

            mu_list.append(mu)

            # print("input", input[:, :, idx].min(), input[:, :, idx].max(), input[:, :, idx].mean())
            # print(f"alp {idx}: ", alp)
            # print(f"mu {idx}: {mu[128, 0]}")

        mu = torch.stack(mu_list, dim=-1)  # [B, 1, T]
        input = input / (mu + eps)
        return input

    @staticmethod
    def forgetting_norm(input, sample_length_in_training):
        """
        Args:
            input: [B, F, T]
            sample_length_in_training:

        Returns:

        """
        assert input.ndim == 3
        batch_size, n_freqs, n_frames = input.size()
        eps = 1e-10
        mu = 0
        alpha = (sample_length_in_training - 1) / (sample_length_in_training + 1)

        mu_list = []
        for idx in range(input.shape[-1]):
            if idx < sample_length_in_training:
                alp = torch.min(torch.tensor([(idx - 1) / (idx + 1), alpha]))
                mu = alp * mu + (1 - alp) * torch.mean(input[:, :, idx], dim=1).reshape(batch_size, 1)  # [B, 1]
            else:
                current_frame_mu = torch.mean(input[:, :, idx], dim=1).reshape(batch_size, 1)  # [B, 1]
                mu = alpha * mu + (1 - alpha) * current_frame_mu

            mu_list.append(mu)

            # print("input", input[:, :, idx].min(), input[:, :, idx].max(), input[:, :, idx].mean())
            # print(f"alp {idx}: ", alp)
            # print(f"mu {idx}: {mu[128, 0]}")

        mu = torch.stack(mu_list, dim=-1)  # [B, 1, T]
        input = input / (mu + eps)
        return input

    @staticmethod
    def hybrid_norm(input, sample_length_in_training=192):
        """
        Args:
            input: [B, F, T]
            sample_length_in_training:

        Returns:
            [B, F, T]
        """
        assert input.ndim == 3
        device = input.device
        data_type = input.dtype
        batch_size, n_freqs, n_frames = input.size()
        eps = 1e-10

        mu = 0
        alpha = (sample_length_in_training - 1) / (sample_length_in_training + 1)
        mu_list = []
        for idx in range(input.shape[-1]):
            if idx < sample_length_in_training:
                alp = torch.min(torch.tensor([(idx - 1) / (idx + 1), alpha]))
                mu = alp * mu + (1 - alp) * torch.mean(input[:, :, idx], dim=1).reshape(batch_size, 1)  # [B, 1]
                mu_list.append(mu)
            else:
                break
        initial_mu = torch.stack(mu_list, dim=-1)  # [B, 1, T]

        step_sum = torch.sum(input, dim=1)  # [B, T]
        cumulative_sum = torch.cumsum(step_sum, dim=-1)  # [B, T]

        entry_count = torch.arange(n_freqs, n_freqs * n_frames + 1, n_freqs, dtype=data_type, device=device)
        entry_count = entry_count.reshape(1, n_frames)  # [1, T]
        entry_count = entry_count.expand_as(cumulative_sum)  # [1, T] => [B, T]

        cum_mean = cumulative_sum / entry_count  # B, T

        cum_mean = cum_mean.reshape(batch_size, 1, n_frames)  # [B, 1, T]

        # print(initial_mu[0, 0, :50])
        # print("-"*60)
        # print(cum_mean[0, 0, :50])
        cum_mean[:, :, :sample_length_in_training] = initial_mu

        return input / (cum_mean + eps)

    @staticmethod
    def offline_laplace_norm(input):
        """

        Args:
            input: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        # utterance-level mu
        mu = torch.mean(input, dim=(1, 2, 3), keepdim=True)

        normed = input / (mu + 1e-5)

        return normed

    @staticmethod
    def cumulative_laplace_norm(input):
        """

        Args:
            input: [B, C, F, T]

        Returns:

        """
        batch_size, num_channels, num_freqs, num_frames = input.size()
        input = input.reshape(batch_size * num_channels, num_freqs, num_frames)

        step_sum = torch.sum(input, dim=1)  # [B * C, F, T] => [B, T]
        cumulative_sum = torch.cumsum(step_sum, dim=-1)  # [B, T]

        entry_count = torch.arange(
            num_freqs,
            num_freqs * num_frames + 1,
            num_freqs,
            dtype=input.dtype,
            device=input.device
        )
        entry_count = entry_count.reshape(1, num_frames)  # [1, T]
        entry_count = entry_count.expand_as(cumulative_sum)  # [1, T] => [B, T]

        cumulative_mean = cumulative_sum / entry_count  # B, T
        cumulative_mean = cumulative_mean.reshape(batch_size * num_channels, 1, num_frames)

        normed = input / (cumulative_mean + EPSILON)

        return normed.reshape(batch_size, num_channels, num_freqs, num_frames)

    @staticmethod
    def offline_gaussian_norm(input):
        """
        Zero-Norm
        Args:
            input: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        mu = torch.mean(input, dim=(1, 2, 3), keepdim=True)
        std = torch.std(input, dim=(1, 2, 3), keepdim=True)

        normed = (input - mu) / (std + 1e-5)

        return normed

    @staticmethod
    def cumulative_layer_norm(input):
        """
        Online zero-norm

        Args:
            input: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        batch_size, num_channels, num_freqs, num_frames = input.size()
        input = input.reshape(batch_size * num_channels, num_freqs, num_frames)

        step_sum = torch.sum(input, dim=1)  # [B * C, F, T] => [B, T]
        step_pow_sum = torch.sum(torch.square(input), dim=1)

        cumulative_sum = torch.cumsum(step_sum, dim=-1)  # [B, T]
        cumulative_pow_sum = torch.cumsum(step_pow_sum, dim=-1)  # [B, T]

        entry_count = torch.arange(
            num_freqs,
            num_freqs * num_frames + 1,
            num_freqs,
            dtype=input.dtype,
            device=input.device
        )
        entry_count = entry_count.reshape(1, num_frames)  # [1, T]
        entry_count = entry_count.expand_as(cumulative_sum)  # [1, T] => [B, T]

        cumulative_mean = cumulative_sum / entry_count  # [B, T]
        cumulative_var = (
                                 cumulative_pow_sum - 2 * cumulative_mean * cumulative_sum) / entry_count + cumulative_mean.pow(
            2)  # [B, T]
        cumulative_std = torch.sqrt(cumulative_var + EPSILON)  # [B, T]

        cumulative_mean = cumulative_mean.reshape(batch_size * num_channels, 1, num_frames)
        cumulative_std = cumulative_std.reshape(batch_size * num_channels, 1, num_frames)

        normed = (input - cumulative_mean) / cumulative_std

        return normed.reshape(batch_size, num_channels, num_freqs, num_frames)

    def norm_wrapper(self, norm_type: str):
        if norm_type == "offline_laplace_norm":
            norm = self.offline_laplace_norm
        elif norm_type == "cumulative_laplace_norm":
            norm = self.cumulative_laplace_norm
        elif norm_type == "offline_gaussian_norm":
            norm = self.offline_gaussian_norm
        elif norm_type == "cumulative_layer_norm":
            norm = self.cumulative_layer_norm
        else:
            raise NotImplementedError("You must set up a type of Norm. "
                                      "e.g. offline_laplace_norm, cumulative_laplace_norm, forgetting_norm, etc.")
        return norm

    def weight_init(self, m):
        """
        Usage:
            model = Model()
            model.apply(weight_init)
        """
        if isinstance(m, nn.Conv1d):
            init.normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.Conv3d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose1d):
            init.normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose3d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.BatchNorm1d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm3d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight.data)
            init.normal_(m.bias.data)
        elif isinstance(m, nn.LSTM):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.LSTMCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.GRU):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.GRUCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)


############################################################################
#                         for data normalization                           #
############################################################################
# get mu and sig
def get_mu_sig(data):
    """Compute mean and standard deviation vector of input data

    Returns:
        mu: mean vector (#dim by one)
        sig: standard deviation vector (#dim by one)
    """
    # Initialize array.
    data_num = len(data)
    mu_utt = []
    tmp_utt = []
    for n in range(data_num):
        dim = len(data[n])
        mu_utt_tmp = np.zeros(dim)
        mu_utt.append(mu_utt_tmp)

        tmp_utt_tmp = np.zeros(dim)
        tmp_utt.append(tmp_utt_tmp)

    # Get mean.
    for n in range(data_num):
        mu_utt[n] = np.mean(data[n], 0)
    mu = mu_utt

    # Get standard deviation.
    for n in range(data_num):
        tmp_utt[n] = np.mean(np.square(data[n] - mu[n]), 0)
    sig = np.sqrt(tmp_utt)

    # Assign unit variance.
    for n in range(len(sig)):
        if sig[n] < 1e-5:
            sig[n] = 1.0
    return np.float16(mu), np.float16(sig)


def get_statistics_inp(inp):
    """Get statistical parameter of input data.

    Args:
        inp: input data

    Returns:
        mu_inp: mean vector of input data
        sig_inp: standard deviation vector of input data
    """

    mu_inp, sig_inp = get_mu_sig(inp)

    return mu_inp, sig_inp


############################################################################
#                       for plotting the samples                           #
############################################################################
def hann_window(win_samp):
    tmp = np.arange(1, win_samp + 1, 1.0, dtype=np.float64)
    window = 0.5 - 0.5 * np.cos((2.0 * np.pi * tmp) / (win_samp + 1))
    return np.float32(window)


def fig2np(fig):
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_spectrogram_to_numpy(input_wav, fs, n_fft, n_overlap, mode, clim, label):
    # cuda to cpu
    input_wav = input_wav.cpu().detach().numpy()

    fig, ax = plt.subplots(figsize=(12, 3))

    if mode == 'phase':
        pxx, freq, t, cax = plt.specgram(input_wav, NFFT=int(n_fft), Fs=int(fs), noverlap=n_overlap,
                                         cmap='jet',
                                         mode=mode)
    else:
        pxx, freq, t, cax = plt.specgram(input_wav, NFFT=int(n_fft), Fs=int(fs), noverlap=n_overlap,
                                         cmap='jet')

    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.clim(clim)

    if label is None:
        fig.colorbar(cax)
    else:
        fig.colorbar(cax, label=label)

    fig.canvas.draw()
    data = fig2np(fig)
    plt.close()
    return data


def plot_mask_to_numpy(mask, fs, n_fft, n_overlap, clim1, clim2, cmap):
    frame_num = mask.shape[0]
    shift_length = n_overlap
    frame_length = n_fft
    signal_length = frame_num * shift_length + frame_length

    xt = np.arange(0, np.floor(10 * signal_length / fs) / 10, step=0.5) / (signal_length / fs) * frame_num + 1e-8
    yt = (n_fft / 2) / (fs / 1000 / 2) * np.arange(0, (fs / 1000 / 2) + 1)

    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(np.transpose(mask), aspect='auto', origin='lower', interpolation='none', cmap=cmap)

    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (kHz)')
    plt.xticks(xt, np.arange(0, np.floor(10 * (signal_length / fs)) / 10, step=0.5))
    plt.yticks(yt, np.int16(np.linspace(0, int((fs / 1000) / 2), len(yt))))
    plt.tight_layout()
    plt.colorbar(im, ax=ax)
    im.set_clim(clim1, clim2)

    fig.canvas.draw()
    data = fig2np(fig)
    plt.close()
    return data


def plot_error_to_numpy(estimated, target, fs, n_fft, n_overlap, mode, clim1, clim2, label):
    fig, ax = plt.subplots(figsize=(12, 3))
    if mode is None:
        pxx1, freq, t, cax = plt.specgram(estimated, NFFT=n_fft, Fs=int(fs), noverlap=n_overlap, cmap='jet')
        pxx2, freq, t, cax = plt.specgram(target, NFFT=n_fft, Fs=int(fs), noverlap=n_overlap, cmap='jet')
        im = ax.imshow(10 * np.log10(pxx1) - 10 * np.log10(pxx2), aspect='auto', origin='lower', interpolation='none',
                       cmap='jet')
    else:
        pxx1, freq, t, cax = plt.specgram(estimated, NFFT=n_fft, Fs=int(fs), noverlap=n_overlap, cmap='jet',
                                          mode=mode)
        pxx2, freq, t, cax = plt.specgram(target, NFFT=n_fft, Fs=int(fs), noverlap=n_overlap, cmap='jet',
                                          mode=mode)
        im = ax.imshow(pxx1 - pxx2, aspect='auto', origin='lower', interpolation='none', cmap='jet')

    frame_num = pxx1.shape[1]
    shift_length = n_overlap
    frame_length = n_fft
    signal_length = frame_num * shift_length + frame_length

    xt = np.arange(0, np.floor(10 * (signal_length / fs)) / 10, step=0.5) / (signal_length / fs) * frame_num
    yt = (n_fft / 2) / (fs / 1000 / 2) * np.arange(0, (fs / 1000 / 2) + 1)

    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (kHz)')
    plt.xticks(xt, np.arange(0, np.floor(10 * (signal_length / fs)) / 10, step=0.5))
    plt.yticks(yt, np.int16(np.linspace(0, int((fs / 1000) / 2), len(yt))))
    plt.tight_layout()
    plt.colorbar(im, ax=ax, label=label)
    im.set_clim(clim1, clim2)

    fig.canvas.draw()
    data = fig2np(fig)
    plt.close()
    return data


############################################################################
#                              for trainer.py                              #
############################################################################
class Bar(object):
    def __init__(self, dataloader):
        if not hasattr(dataloader, 'dataset'):
            raise ValueError('Attribute `dataset` not exists in dataloder.')
        if not hasattr(dataloader, 'batch_size'):
            raise ValueError('Attribute `batch_size` not exists in dataloder.')

        self.dataloader = dataloader
        self.iterator = iter(dataloader)
        self.dataset = dataloader.dataset
        self.batch_size = dataloader.batch_size
        self._idx = 0
        self._batch_idx = 0
        self._time = []
        self._DISPLAY_LENGTH = 50

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._time) < 2:
            self._time.append(time.time())

        self._batch_idx += self.batch_size
        if self._batch_idx > len(self.dataset):
            self._batch_idx = len(self.dataset)

        try:
            batch = next(self.iterator)
            self._display()
        except StopIteration:
            raise StopIteration()

        self._idx += 1
        if self._idx >= len(self.dataloader):
            self._reset()

        return batch

    def _display(self):
        if len(self._time) > 1:
            t = (self._time[-1] - self._time[-2])
            eta = t * (len(self.dataloader) - self._idx)
        else:
            eta = 0

        rate = self._idx / len(self.dataloader)
        len_bar = int(rate * self._DISPLAY_LENGTH)
        bar = ('=' * len_bar + '>').ljust(self._DISPLAY_LENGTH, '.')
        idx = str(self._batch_idx).rjust(len(str(len(self.dataset))), ' ')

        tmpl = '\r{}/{}: [{}] - ETA {:.1f}s'.format(
            idx,
            len(self.dataset),
            bar,
            eta
        )
        print(tmpl, end='')
        if self._batch_idx == len(self.dataset):
            print()

    def _reset(self):
        self._idx = 0
        self._batch_idx = 0
        self._time = []
