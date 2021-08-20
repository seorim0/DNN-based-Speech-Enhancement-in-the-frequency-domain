import torch
import torch.nn as nn
import torch.nn.functional as F
from tools_for_model import ConvSTFT, ConviSTFT, cPReLU, \
    ComplexConv2d, ComplexConvTranspose2d, NavieComplexLSTM, complex_cat, ComplexBatchNorm, LayerNorm, GroupNorm2d
import config as cfg
from tools_for_loss import sdr, sdr_linear, si_sdr, si_snr, get_array_lms_loss, pmsqe_stft, pmsqe, get_pam_loss
from asteroid_filterbanks import transforms
from scipy.io.wavfile import write as wav_write
import matplotlib.pyplot as plt


class complex_model(nn.Module):

    def __init__(
            self,
            rnn_layers=cfg.rnn_layers,
            rnn_units=cfg.rnn_units,
            win_len=cfg.win_len,
            win_inc=cfg.win_inc,
            fft_len=cfg.fft_len,
            win_type=cfg.window,
            masking_mode=None if cfg.masking_mode == 'Direct(None make)' else cfg.masking_mode,
            use_cbn=True if cfg.batch_norm == 'complex' else False,
            kernel_size=5
    ):
        '''
            rnn_layers: the number of lstm layers in the crn,
            rnn_units: for clstm, rnn_units = real+imag
        '''

        super(complex_model, self).__init__()

        # for fft
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.win_type = win_type

        input_dim = win_len
        output_dim = win_len

        self.rnn_units = rnn_units
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = rnn_layers
        self.kernel_size = kernel_size
        kernel_num = cfg.dccrn_kernel_num
        self.kernel_num = [2] + kernel_num
        self.masking_mode = masking_mode

        # bidirectional=True
        bidirectional = False
        fac = 2 if bidirectional else 1

        fix = True
        self.fix = fix
        self.stft = ConvSTFT(self.win_len, self.win_inc, fft_len, self.win_type, 'complex', fix=fix)
        self.istft = ConviSTFT(self.win_len, self.win_inc, fft_len, self.win_type, 'complex', fix=fix)

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for idx in range(len(self.kernel_num) - 1):
            self.encoder.append(
                nn.Sequential(
                    # nn.ConstantPad2d([0, 0, 0, 0], 0),
                    ComplexConv2d(
                        self.kernel_num[idx],
                        self.kernel_num[idx + 1],
                        kernel_size=(self.kernel_size, 2),
                        stride=(2, 1),
                        padding=(2, 1)
                    ),
                    #GroupNorm2d(self.kernel_num[idx + 1]) if not use_cbn else ComplexBatchNorm(
                    #nn.GroupNorm(32,self.kernel_num[idx + 1]) if not use_cbn else ComplexBatchNorm(
                    #LayerNorm(self.kernel_num[idx + 1]) if not use_cbn else ComplexBatchNorm(
                    #nn.InstanceNorm2d(self.kernel_num[idx + 1]) if not use_cbn else ComplexBatchNorm(
                    nn.BatchNorm2d(self.kernel_num[idx + 1]) if not use_cbn else ComplexBatchNorm(
                        self.kernel_num[idx + 1]),
                    # nn.ELU()
                    nn.PReLU()  ##
                    # nn.ReLU()
                    # cPReLU()
                )
            )
        hidden_dim = self.fft_len // (2 ** (len(self.kernel_num)))

        if cfg.lstm == 'complex':
            rnns = []
            for idx in range(rnn_layers):
                rnns.append(
                    NavieComplexLSTM(
                        input_size=hidden_dim * self.kernel_num[-1] if idx == 0 else self.rnn_units,
                        hidden_size=self.rnn_units,
                        bidirectional=bidirectional,
                        batch_first=False,
                        projection_dim=hidden_dim * self.kernel_num[-1] if idx == rnn_layers - 1 else None,
                    )
                )
                self.enhance = nn.Sequential(*rnns)
        else:
            self.enhance = nn.LSTM(
                input_size=hidden_dim * self.kernel_num[-1],
                hidden_size=self.rnn_units,
                num_layers=2,
                dropout=0.0,
                bidirectional=bidirectional,
                batch_first=False
            )
            self.tranform = nn.Linear(self.rnn_units * fac, hidden_dim * self.kernel_num[-1])

        if cfg.skip_type:
            for idx in range(len(self.kernel_num) - 1, 0, -1):
                if idx != 1:
                    self.decoder.append(
                        nn.Sequential(
                            ComplexConvTranspose2d(
                                self.kernel_num[idx] * 2,
                                self.kernel_num[idx - 1],
                                kernel_size=(self.kernel_size, 2),
                                stride=(2, 1),
                                padding=(2, 0),
                                output_padding=(1, 0)
                            ),
                            #GroupNorm2d(self.kernel_num[idx - 1]) if not use_cbn else ComplexBatchNorm(
                            #LayerNorm(self.kernel_num[idx - 1]) if not use_cbn else ComplexBatchNorm(
                            #nn.InstanceNorm2d(self.kernel_num[idx - 1]) if not use_cbn else ComplexBatchNorm(
                            #nn.GroupNorm(32, self.kernel_num[idx - 1]) if not use_cbn else ComplexBatchNorm(
                            nn.BatchNorm2d(self.kernel_num[idx - 1]) if not use_cbn else ComplexBatchNorm(
                                self.kernel_num[idx - 1]),
                            # nn.ELU()
                            nn.PReLU()  ##
                            # nn.ReLU()
                            # cPReLU()
                        )
                    )
                else:
                    self.decoder.append(
                        nn.Sequential(
                            ComplexConvTranspose2d(
                                self.kernel_num[idx] * 2,
                                self.kernel_num[idx - 1],
                                kernel_size=(self.kernel_size, 2),
                                stride=(2, 1),
                                padding=(2, 0),
                                output_padding=(1, 0)
                            ),
                        )
                    )
        else:
            for idx in range(len(self.kernel_num) - 1, 0, -1):
                if idx != 1:
                    self.decoder.append(
                        nn.Sequential(
                            ComplexConvTranspose2d(
                                self.kernel_num[idx],
                                self.kernel_num[idx - 1],
                                kernel_size=(self.kernel_size, 2),
                                stride=(2, 1),
                                padding=(2, 0),
                                output_padding=(1, 0)
                            ),
                            nn.BatchNorm2d(self.kernel_num[idx - 1]) if not use_cbn else ComplexBatchNorm(
                                self.kernel_num[idx - 1]),
                            # nn.ELU()
                            nn.PReLU()
                        )
                    )
                else:
                    self.decoder.append(
                        nn.Sequential(
                            ComplexConvTranspose2d(
                                self.kernel_num[idx],
                                self.kernel_num[idx - 1],
                                kernel_size=(self.kernel_size, 2),
                                stride=(2, 1),
                                padding=(2, 0),
                                output_padding=(1, 0)
                            ),
                        )
                    )
        self.flatten_parameters()

    def flatten_parameters(self):
        if isinstance(self.enhance, nn.LSTM):
            self.enhance.flatten_parameters()

    def forward(self, inputs, direct_mapping=False):
        specs = self.stft(inputs)
        real = specs[:, :self.fft_len // 2 + 1]
        imag = specs[:, self.fft_len // 2 + 1:]
        spec_mags = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)

        spec_phase = torch.atan2(imag, real)
        spec_phase = spec_phase
        cspecs = torch.stack([real, imag], 1)
        cspecs = cspecs[:, :, 1:]
        '''
        means = torch.mean(cspecs, [1,2,3], keepdim=True)
        std = torch.std(cspecs, [1,2,3], keepdim=True )
        normed_cspecs = (cspecs-means)/(std+1e-8)
        out = normed_cspecs
        '''

        out = cspecs
        encoder_out = []

        for idx, layer in enumerate(self.encoder):
            out = layer(out)
            #    print('encoder', out.size())
            encoder_out.append(out)

        batch_size, channels, dims, lengths = out.size()
        out = out.permute(3, 0, 1, 2)
        if cfg.lstm == 'complex':
            r_rnn_in = out[:, :, :channels // 2]
            i_rnn_in = out[:, :, channels // 2:]
            r_rnn_in = torch.reshape(r_rnn_in, [lengths, batch_size, channels // 2 * dims])
            i_rnn_in = torch.reshape(i_rnn_in, [lengths, batch_size, channels // 2 * dims])

            r_rnn_in, i_rnn_in = self.enhance([r_rnn_in, i_rnn_in])

            r_rnn_in = torch.reshape(r_rnn_in, [lengths, batch_size, channels // 2, dims])
            i_rnn_in = torch.reshape(i_rnn_in, [lengths, batch_size, channels // 2, dims])
            out = torch.cat([r_rnn_in, i_rnn_in], 2)
        else:
            # to [L, B, C, D]
            out = torch.reshape(out, [lengths, batch_size, channels * dims])
            out, _ = self.enhance(out)
            out = self.tranform(out)
            out = torch.reshape(out, [lengths, batch_size, channels, dims])

        out = out.permute(1, 2, 3, 0)

        if cfg.skip_type:  # use skip connection
            for idx in range(len(self.decoder)):
                out = complex_cat([out, encoder_out[-1 - idx]], 1)
                out = self.decoder[idx](out)
                out = out[..., 1:]  #
        else:
            for idx in range(len(self.decoder)):
                out = self.decoder[idx](out)
                out = out[..., 1:]

        if direct_mapping:  # for direct mapping model or Cyclic model
            out_real = out[:, 0]
            out_imag = out[:, 1]
            out_real = F.pad(out_real, [0, 0, 1, 0])
            out_imag = F.pad(out_imag, [0, 0, 1, 0])

            # a = real*out_real
            # b = imag*out_imag
            # for i in range(483):
            #     fig = plt.figure()
            #     ax = fig.add_subplot(1, 1, 1)
            #     r = real[0]
            #     r = r.permute(1, 0)
            #     r = r.cpu().detach().numpy()
            #     ax.plot(r[100+i], lw=2)
            #
            #     fig = plt.figure()
            #     ax0 = fig.add_subplot(1, 1, 1)
            #     r = targets_real[0]
            #     r = r.permute(1, 0)
            #     r = r.cpu().detach().numpy()
            #     ax0.plot(r[100+i], lw=2)
            #
            #     fig = plt.figure()
            #     ax00 = fig.add_subplot(1, 1, 1)
            #     r = out_real[0]
            #     r = r.permute(1, 0)
            #     r = r.cpu().detach().numpy()
            #     ax00.plot(r[100+i], lw=2)
            #
            #     fig = plt.figure()
            #     ax000 = fig.add_subplot(1, 1, 1)
            #     r = a[0]
            #     r = r.permute(1, 0)
            #     r = r.cpu().detach().numpy()
            #     ax000.plot(r[100+i], lw=2)

            # out_mags = (out_real ** 2 + out_imag ** 2) ** 0.5
            # real_phase = out_real / (out_mags + 1e-8)
            # imag_phase = out_imag / (out_mags + 1e-8)
            # out_phase = torch.atan2(
            #     imag_phase,
            #     real_phase
            # )
            #
            # out_mags = torch.tanh(out_mags)
            # # out_phase = spec_phase + out_phase
            # out_real = out_mags * torch.cos(out_phase)
            # out_imag = out_mags * torch.sin(out_phase)
            out_real = torch.where(real == 0, out_real * real, out_real)
            out_imag = torch.where(imag == 0, out_imag * imag, out_imag)

            out_spec = torch.cat([out_real, out_imag], 1)

            # # 이전
            # out_real = out[:, 0]
            # out_imag = out[:, 1]
            # out_spec = torch.cat([out_real, out_imag], 1)
            # out_spec = F.pad(out_spec, [0, 0, 2, 0])
            # # out_real = F.pad(out_real, [0, 0, 1, 0])
            # # out_imag = F.pad(out_imag, [0, 0, 1, 0])  # 훈련할 때 어디에 padding 하는지에 따라서 (est)pesq 값 너무 달라짐
            # # target_specs = self.stft(targets)
            # # out_spec = torch.cat([out_real, out_imag], 1)
            # # return 0, 0, out_spec, target_specs
            # # # ##################################################
            # # spec_mags = torch.sqrt(out_real ** 2 + out_imag ** 2 + 1e-8)
            # # spec_phase = torch.atan2(out_imag, out_real)
            # # out_real = spec_mags * torch.cos(spec_phase)
            # # out_imag = spec_mags * torch.sin(spec_phase)
            #
            # # # out_mags = (out_real ** 2 + out_imag ** 2) ** 0.5
            # # # real_phase = out_real / (out_mags + 1e-8)
            # # # imag_phase = out_imag / (out_mags + 1e-8)
            # # # out_phase = torch.atan2(
            # # #     imag_phase,
            # # #     real_phase
            # # # )
            # # #
            # # # # mask_mags = torch.clamp_(mask_mags,0,100)
            # # # out_mags = torch.tanh(out_mags)
            # # # out_real = out_mags * torch.cos(out_phase)
            # # # out_imag = out_mags * torch.sin(out_phase)
            # # # ##################################################
            # # out_m_real, out_m_imag = real * out_real, imag * out_imag
            # #
            # # for i in range(len(out_real[0])):
            # #     # figure 1
            # #     fig = plt.figure()
            # #     ax1 = fig.add_subplot(1, 1, 1)
            # #     r = out_real.permute(0, 2, 1)
            # #     im = out_imag.permute(0, 2, 1)
            # #     r = r[0][i].cpu().detach().numpy()
            # #     im = im[0][i].cpu().detach().numpy()
            # #     ax1.plot(r, lw=2)
            # #     ax1.plot(im, lw=2)
            # #
            # #     fig = plt.figure()
            # #     ax1 = fig.add_subplot(1, 1, 1)
            # #     r = out_m_real.permute(0, 2, 1)
            # #     im = out_m_imag.permute(0, 2, 1)
            # #     r = r[0][i].cpu().detach().numpy()
            # #     im = im[0][i].cpu().detach().numpy()
            # #     ax1.plot(r, lw=2)
            # #     ax1.plot(im, lw=2)
        else:
            #    print('decoder', out.size())
            mask_real = out[:, 0]
            mask_imag = out[:, 1]
            mask_real = F.pad(mask_real, [0, 0, 1, 0])
            mask_imag = F.pad(mask_imag, [0, 0, 1, 0])

            if self.masking_mode == 'E':
                mask_mags = (mask_real ** 2 + mask_imag ** 2) ** 0.5
                real_phase = mask_real / (mask_mags + 1e-8)
                imag_phase = mask_imag / (mask_mags + 1e-8)
                mask_phase = torch.atan2(
                    imag_phase,
                    real_phase
                )

                # mask_mags = torch.clamp_(mask_mags,0,100)
                mask_mags = torch.tanh(mask_mags)
                est_mags = mask_mags * spec_mags
                est_phase = spec_phase + mask_phase
                out_real = est_mags * torch.cos(est_phase)
                out_imag = est_mags * torch.sin(est_phase)
            elif self.masking_mode == 'C':
                out_real, out_imag = real * mask_real - imag * mask_imag, real * mask_imag + imag * mask_real
            elif self.masking_mode == 'R':
                out_real, out_imag = real * mask_real, imag * mask_imag

            out_spec = torch.cat([out_real, out_imag], 1)

        out_wav = self.istft(out_spec)
        out_wav = torch.squeeze(out_wav, 1)
        out_wav = torch.clamp_(out_wav, -1, 1)

        # estimated_wavs = out_wav.cpu().detach().numpy()
        # for i in range(len(estimated_wavs)):
        #     wav_write('./test/{}.wav'.format(i), cfg.fs, estimated_wavs[i])

        return out_real, out_imag, out_wav

    def get_params(self, weight_decay=0.0):
        # add L2 penalty
        weights, biases = [], []
        for name, param in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{
            'params': weights,
            'weight_decay': weight_decay,
        }, {
            'params': biases,
            'weight_decay': 0.0,
        }]
        return params

    def loss(self, estimated, target, real_spec=0, img_spec=0, GMT=0, perceptual=False):
        if perceptual:
            if cfg.perceptual == 'LMS':
                # for lms loss calculation
                clean_specs = self.stft(target)
                clean_real = clean_specs[:, :self.fft_len // 2 + 1]
                clean_imag = clean_specs[:, self.fft_len // 2 + 1:]
                clean_mags = torch.sqrt(clean_real ** 2 + clean_imag ** 2 + 1e-7)

                est_clean_mags = torch.sqrt(real_spec ** 2 + img_spec ** 2 + 1e-7)
                lms_loss = get_array_lms_loss(clean_mags, est_clean_mags)
                return lms_loss
            elif cfg.perceptual == 'PMSQE':
                ref_wav = target.reshape(-1, 3, 16000)  # dataset data shape
                est_wav = estimated.reshape(-1, 3, 16000)
                ref_wav = ref_wav.cpu()
                est_wav = est_wav.cpu()

                ref_spec = transforms.take_mag(pmsqe_stft(ref_wav))
                est_spec = transforms.take_mag(pmsqe_stft(est_wav))

                loss = pmsqe(ref_spec, est_spec)
                loss = loss.cuda()
                return loss
            elif cfg.perceptual == 'PAM':
                residual_noise = estimated - target
                # noise_spec = self.stft(residual_noise)
                noise_spec = self.stft(residual_noise/512)
                noise_real = noise_spec[:, : self.fft_len // 2 + 1]
                noise_img = noise_spec[:, self.fft_len // 2 + 1:]

                noise_mag = torch.sqrt(noise_real ** 2 + noise_img ** 2 + 1e-7)

                pam_loss = get_pam_loss(noise_mag, GMT)
                return pam_loss
        else:
            if cfg.loss == 'MSE':
                return F.mse_loss(estimated, target, reduction='mean')
            elif cfg.loss == 'SDR':
                return -sdr(target, estimated)
            elif cfg.loss == 'SI-SNR':
                return -(si_snr(estimated, target))
            elif cfg.loss == 'SI-SDR':
                return -(si_sdr(target, estimated))


class complex_model_direct(nn.Module):

    def __init__(
            self,
            rnn_layers=cfg.rnn_layers,
            rnn_units=cfg.rnn_units,
            win_len=cfg.win_len,
            win_inc=cfg.win_inc,
            fft_len=cfg.fft_len,
            win_type=cfg.window,
            use_cbn=True if cfg.batch_norm == 'complex' else False,
            kernel_size=5
    ):
        '''
            rnn_layers: the number of lstm layers in the crn,
            rnn_units: for clstm, rnn_units = real+imag
        '''

        super(complex_model_direct, self).__init__()

        # for fft
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.win_type = win_type

        input_dim = win_len
        output_dim = win_len

        self.rnn_units = rnn_units
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = rnn_layers
        self.kernel_size = kernel_size
        kernel_num = cfg.dccrn_kernel_num
        self.kernel_num = [2] + kernel_num

        # bidirectional=True
        bidirectional = False
        fac = 2 if bidirectional else 1

        fix = True
        self.fix = fix
        self.stft = ConvSTFT(self.win_len, self.win_inc, fft_len, self.win_type, 'complex', fix=fix)
        self.istft = ConviSTFT(self.win_len, self.win_inc, fft_len, self.win_type, 'complex', fix=fix)

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for idx in range(len(self.kernel_num) - 1):
            self.encoder.append(
                nn.Sequential(
                    # nn.ConstantPad2d([0, 0, 0, 0], 0),
                    ComplexConv2d(
                        self.kernel_num[idx],
                        self.kernel_num[idx + 1],
                        kernel_size=(self.kernel_size, 2),
                        stride=(2, 1),
                        padding=(2, 1)
                    ),
                    #GroupNorm2d(self.kernel_num[idx + 1]) if not use_cbn else ComplexBatchNorm(
                    #nn.GroupNorm(32,self.kernel_num[idx + 1]) if not use_cbn else ComplexBatchNorm(
                    #LayerNorm(self.kernel_num[idx + 1]) if not use_cbn else ComplexBatchNorm(
                    #nn.InstanceNorm2d(self.kernel_num[idx + 1]) if not use_cbn else ComplexBatchNorm(
                    nn.BatchNorm2d(self.kernel_num[idx + 1]) if not use_cbn else ComplexBatchNorm(
                        self.kernel_num[idx + 1]),
                    # nn.ELU()
                    nn.PReLU()  ##
                    # nn.ReLU()
                    # cPReLU()
                )
            )
        hidden_dim = self.fft_len // (2 ** (len(self.kernel_num)))

        if cfg.lstm == 'complex':
            rnns = []
            for idx in range(rnn_layers):
                rnns.append(
                    NavieComplexLSTM(
                        input_size=hidden_dim * self.kernel_num[-1] if idx == 0 else self.rnn_units,
                        hidden_size=self.rnn_units,
                        bidirectional=bidirectional,
                        batch_first=False,
                        projection_dim=hidden_dim * self.kernel_num[-1] if idx == rnn_layers - 1 else None,
                    )
                )
                self.enhance = nn.Sequential(*rnns)
        else:
            self.enhance = nn.LSTM(
                input_size=hidden_dim * self.kernel_num[-1],
                hidden_size=self.rnn_units,
                num_layers=2,
                dropout=0.0,
                bidirectional=bidirectional,
                batch_first=False
            )
            self.tranform = nn.Linear(self.rnn_units * fac, hidden_dim * self.kernel_num[-1])

        if cfg.skip_type:
            for idx in range(len(self.kernel_num) - 1, 0, -1):
                if idx != 1:
                    self.decoder.append(
                        nn.Sequential(
                            ComplexConvTranspose2d(
                                self.kernel_num[idx] * 2,
                                self.kernel_num[idx - 1],
                                kernel_size=(self.kernel_size, 2),
                                stride=(2, 1),
                                padding=(2, 0),
                                output_padding=(1, 0)
                            ),
                            #GroupNorm2d(self.kernel_num[idx - 1]) if not use_cbn else ComplexBatchNorm(
                            #LayerNorm(self.kernel_num[idx - 1]) if not use_cbn else ComplexBatchNorm(
                            #nn.InstanceNorm2d(self.kernel_num[idx - 1]) if not use_cbn else ComplexBatchNorm(
                            #nn.GroupNorm(32, self.kernel_num[idx - 1]) if not use_cbn else ComplexBatchNorm(
                            nn.BatchNorm2d(self.kernel_num[idx - 1]) if not use_cbn else ComplexBatchNorm(
                                self.kernel_num[idx - 1]),
                            # nn.ELU()
                            nn.PReLU()  ##
                            # nn.ReLU()
                            # cPReLU()
                        )
                    )
                else:
                    self.decoder.append(
                        nn.Sequential(
                            ComplexConvTranspose2d(
                                self.kernel_num[idx] * 2,
                                self.kernel_num[idx - 1],
                                kernel_size=(self.kernel_size, 2),
                                stride=(2, 1),
                                padding=(2, 0),
                                output_padding=(1, 0)
                            ),
                        )
                    )
        else:
            for idx in range(len(self.kernel_num) - 1, 0, -1):
                if idx != 1:
                    self.decoder.append(
                        nn.Sequential(
                            ComplexConvTranspose2d(
                                self.kernel_num[idx],
                                self.kernel_num[idx - 1],
                                kernel_size=(self.kernel_size, 2),
                                stride=(2, 1),
                                padding=(2, 0),
                                output_padding=(1, 0)
                            ),
                            nn.BatchNorm2d(self.kernel_num[idx - 1]) if not use_cbn else ComplexBatchNorm(
                                self.kernel_num[idx - 1]),
                            # nn.ELU()
                            nn.PReLU()
                        )
                    )
                else:
                    self.decoder.append(
                        nn.Sequential(
                            ComplexConvTranspose2d(
                                self.kernel_num[idx],
                                self.kernel_num[idx - 1],
                                kernel_size=(self.kernel_size, 2),
                                stride=(2, 1),
                                padding=(2, 0),
                                output_padding=(1, 0)
                            ),
                        )
                    )
        self.flatten_parameters()

    def flatten_parameters(self):
        if isinstance(self.enhance, nn.LSTM):
            self.enhance.flatten_parameters()

    def forward(self, inputs, targets):
        targets_specs = self.stft(targets)
        targets_real = targets_specs[:, :self.fft_len // 2 + 1]
        targets_imag = targets_specs[:, self.fft_len // 2 + 1:]

        specs = self.stft(inputs)
        real = specs[:, :self.fft_len // 2 + 1]
        imag = specs[:, self.fft_len // 2 + 1:]

        cspecs = torch.stack([real, imag], 1)
        cspecs = cspecs[:, :, 1:]
        '''
        means = torch.mean(cspecs, [1,2,3], keepdim=True)
        std = torch.std(cspecs, [1,2,3], keepdim=True )
        normed_cspecs = (cspecs-means)/(std+1e-8)
        out = normed_cspecs
        '''

        out = cspecs
        encoder_out = []

        for idx, layer in enumerate(self.encoder):
            out = layer(out)
            #    print('encoder', out.size())
            encoder_out.append(out)

        batch_size, channels, dims, lengths = out.size()
        out = out.permute(3, 0, 1, 2)
        if cfg.lstm == 'complex':
            r_rnn_in = out[:, :, :channels // 2]
            i_rnn_in = out[:, :, channels // 2:]
            r_rnn_in = torch.reshape(r_rnn_in, [lengths, batch_size, channels // 2 * dims])
            i_rnn_in = torch.reshape(i_rnn_in, [lengths, batch_size, channels // 2 * dims])

            r_rnn_in, i_rnn_in = self.enhance([r_rnn_in, i_rnn_in])

            r_rnn_in = torch.reshape(r_rnn_in, [lengths, batch_size, channels // 2, dims])
            i_rnn_in = torch.reshape(i_rnn_in, [lengths, batch_size, channels // 2, dims])
            out = torch.cat([r_rnn_in, i_rnn_in], 2)
        else:
            # to [L, B, C, D]
            out = torch.reshape(out, [lengths, batch_size, channels * dims])
            out, _ = self.enhance(out)
            out = self.tranform(out)
            out = torch.reshape(out, [lengths, batch_size, channels, dims])

        out = out.permute(1, 2, 3, 0)

        if cfg.skip_type:  # use skip connection
            for idx in range(len(self.decoder)):
                out = complex_cat([out, encoder_out[-1 - idx]], 1)
                out = self.decoder[idx](out)
                out = out[..., 1:]  #
        else:
            for idx in range(len(self.decoder)):
                out = self.decoder[idx](out)
                out = out[..., 1:]


        out_real = out[:, 0]
        out_imag = out[:, 1]
        out_real = F.pad(out_real, [0, 0, 1, 0])
        out_imag = F.pad(out_imag, [0, 0, 1, 0])

        # a = real*out_real
            # b = imag*out_imag
            # for i in range(483):
            #     fig = plt.figure()
            #     ax = fig.add_subplot(1, 1, 1)
            #     r = real[0]
            #     r = r.permute(1, 0)
            #     r = r.cpu().detach().numpy()
            #     ax.plot(r[100+i], lw=2)
            #
            #     fig = plt.figure()
            #     ax0 = fig.add_subplot(1, 1, 1)
            #     r = targets_real[0]
            #     r = r.permute(1, 0)
            #     r = r.cpu().detach().numpy()
            #     ax0.plot(r[100+i], lw=2)
            #
            #     fig = plt.figure()
            #     ax00 = fig.add_subplot(1, 1, 1)
            #     r = out_real[0]
            #     r = r.permute(1, 0)
            #     r = r.cpu().detach().numpy()
            #     ax00.plot(r[100+i], lw=2)
            #
            #     fig = plt.figure()
            #     ax000 = fig.add_subplot(1, 1, 1)
            #     r = a[0]
            #     r = r.permute(1, 0)
            #     r = r.cpu().detach().numpy()
            #     ax000.plot(r[100+i], lw=2)

            # out_mags = (out_real ** 2 + out_imag ** 2) ** 0.5
            # real_phase = out_real / (out_mags + 1e-8)
            # imag_phase = out_imag / (out_mags + 1e-8)
            # out_phase = torch.atan2(
            #     imag_phase,
            #     real_phase
            # )
            #
            # out_mags = torch.tanh(out_mags)
            # # out_phase = spec_phase + out_phase
            # out_real = out_mags * torch.cos(out_phase)
            # out_imag = out_mags * torch.sin(out_phase)
        out_real = torch.where(real == 0, out_real * real, out_real)
        out_imag = torch.where(imag == 0, out_imag * imag, out_imag)

        out_spec = torch.cat([out_real, out_imag], 1)

        # # 이전
            # out_real = out[:, 0]
            # out_imag = out[:, 1]
            # out_spec = torch.cat([out_real, out_imag], 1)
            # out_spec = F.pad(out_spec, [0, 0, 2, 0])
            # # out_real = F.pad(out_real, [0, 0, 1, 0])
            # # out_imag = F.pad(out_imag, [0, 0, 1, 0])  # 훈련할 때 어디에 padding 하는지에 따라서 (est)pesq 값 너무 달라짐
            # # target_specs = self.stft(targets)
            # # out_spec = torch.cat([out_real, out_imag], 1)
            # # return 0, 0, out_spec, target_specs
            # # # ##################################################
            # # spec_mags = torch.sqrt(out_real ** 2 + out_imag ** 2 + 1e-8)
            # # spec_phase = torch.atan2(out_imag, out_real)
            # # out_real = spec_mags * torch.cos(spec_phase)
            # # out_imag = spec_mags * torch.sin(spec_phase)
            #
            # # # out_mags = (out_real ** 2 + out_imag ** 2) ** 0.5
            # # # real_phase = out_real / (out_mags + 1e-8)
            # # # imag_phase = out_imag / (out_mags + 1e-8)
            # # # out_phase = torch.atan2(
            # # #     imag_phase,
            # # #     real_phase
            # # # )
            # # #
            # # # # mask_mags = torch.clamp_(mask_mags,0,100)
            # # # out_mags = torch.tanh(out_mags)
            # # # out_real = out_mags * torch.cos(out_phase)
            # # # out_imag = out_mags * torch.sin(out_phase)
            # # # ##################################################
            # # out_m_real, out_m_imag = real * out_real, imag * out_imag
            # #
            # # for i in range(len(out_real[0])):
            # #     # figure 1
            # #     fig = plt.figure()
            # #     ax1 = fig.add_subplot(1, 1, 1)
            # #     r = out_real.permute(0, 2, 1)
            # #     im = out_imag.permute(0, 2, 1)
            # #     r = r[0][i].cpu().detach().numpy()
            # #     im = im[0][i].cpu().detach().numpy()
            # #     ax1.plot(r, lw=2)
            # #     ax1.plot(im, lw=2)
            # #
            # #     fig = plt.figure()
            # #     ax1 = fig.add_subplot(1, 1, 1)
            # #     r = out_m_real.permute(0, 2, 1)
            # #     im = out_m_imag.permute(0, 2, 1)
            # #     r = r[0][i].cpu().detach().numpy()
            # #     im = im[0][i].cpu().detach().numpy()
            # #     ax1.plot(r, lw=2)
            # #     ax1.plot(im, lw=2)

        out_wav = self.istft(out_spec)
        out_wav = torch.squeeze(out_wav, 1)
        out_wav = torch.clamp_(out_wav, -1, 1)

        # estimated_wavs = out_wav.cpu().detach().numpy()
        # for i in range(len(estimated_wavs)):
        #     wav_write('./test/{}.wav'.format(i), cfg.fs, estimated_wavs[i])

        return out_real, targets_real, out_imag, targets_imag, out_wav

    def get_params(self, weight_decay=0.0):
        # add L2 penalty
        weights, biases = [], []
        for name, param in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{
            'params': weights,
            'weight_decay': weight_decay,
        }, {
            'params': biases,
            'weight_decay': 0.0,
        }]
        return params

    def loss(self, estimated, target, real_spec=0, img_spec=0, GMT=0, perceptual=False):
        if perceptual:
            if cfg.perceptual == 'LMS':
                # for lms loss calculation
                clean_specs = self.stft(target)
                clean_real = clean_specs[:, :self.fft_len // 2 + 1]
                clean_imag = clean_specs[:, self.fft_len // 2 + 1:]
                clean_mags = torch.sqrt(clean_real ** 2 + clean_imag ** 2 + 1e-7)

                est_clean_mags = torch.sqrt(real_spec ** 2 + img_spec ** 2 + 1e-7)
                lms_loss = get_array_lms_loss(clean_mags, est_clean_mags)
                return lms_loss
            elif cfg.perceptual == 'PMSQE':
                ref_wav = target.reshape(-1, 3, 16000)  # dataset data shape
                est_wav = estimated.reshape(-1, 3, 16000)
                ref_wav = ref_wav.cpu()
                est_wav = est_wav.cpu()

                ref_spec = transforms.take_mag(pmsqe_stft(ref_wav))
                est_spec = transforms.take_mag(pmsqe_stft(est_wav))

                loss = pmsqe(ref_spec, est_spec)
                loss = loss.cuda()
                return loss
            elif cfg.perceptual == 'PAM':
                residual_noise = estimated - target
                # noise_spec = self.stft(residual_noise)
                noise_spec = self.stft(residual_noise/512)
                noise_real = noise_spec[:, : self.fft_len // 2 + 1]
                noise_img = noise_spec[:, self.fft_len // 2 + 1:]

                noise_mag = torch.sqrt(noise_real ** 2 + noise_img ** 2 + 1e-7)

                pam_loss = get_pam_loss(noise_mag, GMT)
                return pam_loss
        else:
            if cfg.loss == 'MSE':
                return F.mse_loss(estimated, target, reduction='mean')
            elif cfg.loss == 'SDR':
                return -sdr(target, estimated)
            elif cfg.loss == 'SI-SNR':
                return -(si_snr(estimated, target))
            elif cfg.loss == 'SI-SDR':
                return -(si_sdr(target, estimated))


class complex_model_latent_vector(nn.Module):

    def __init__(
            self,
            rnn_layers=cfg.rnn_layers,
            rnn_units=cfg.rnn_units,
            win_len=cfg.win_len,
            win_inc=cfg.win_inc,
            fft_len=cfg.fft_len,
            win_type=cfg.window,
            masking_mode=None if cfg.masking_mode == 'Direct(None make)' else cfg.masking_mode,
            use_cbn=True if cfg.batch_norm == 'complex' else False,
            kernel_size=5
    ):
        '''
            rnn_layers: the number of lstm layers in the crn,
            rnn_units: for clstm, rnn_units = real+imag
        '''

        super(complex_model_latent_vector, self).__init__()

        # for fft
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.win_type = win_type

        input_dim = win_len
        output_dim = win_len

        self.rnn_units = rnn_units
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = rnn_layers
        self.kernel_size = kernel_size
        kernel_num = cfg.dccrn_kernel_num
        self.kernel_num = [2] + kernel_num
        self.masking_mode = masking_mode

        # bidirectional=True
        bidirectional = False
        fac = 2 if bidirectional else 1

        fix = True
        self.fix = fix
        self.stft = ConvSTFT(self.win_len, self.win_inc, fft_len, self.win_type, 'complex', fix=fix)
        self.istft = ConviSTFT(self.win_len, self.win_inc, fft_len, self.win_type, 'complex', fix=fix)

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for idx in range(len(self.kernel_num) - 1):
            self.encoder.append(
                nn.Sequential(
                    # nn.ConstantPad2d([0, 0, 0, 0], 0),
                    ComplexConv2d(
                        self.kernel_num[idx],
                        self.kernel_num[idx + 1],
                        kernel_size=(self.kernel_size, 2),
                        stride=(2, 1),
                        padding=(2, 1)
                    ),
                    nn.BatchNorm2d(self.kernel_num[idx + 1]) if not use_cbn else ComplexBatchNorm(
                        self.kernel_num[idx + 1]),
                    nn.PReLU()
                )
            )
        hidden_dim = self.fft_len // (2 ** (len(self.kernel_num)))

        if cfg.lstm == 'complex':
            rnn1 = []
            rnn2 = []
            rnn1.append(
                    NavieComplexLSTM(
                        input_size=hidden_dim * self.kernel_num[-1],
                        hidden_size=self.rnn_units,
                        bidirectional=bidirectional,
                        batch_first=False,
                        projection_dim=None,
                    )
                )
            rnn2.append(
                NavieComplexLSTM(
                    input_size=self.rnn_units,
                    hidden_size=self.rnn_units,
                    bidirectional=bidirectional,
                    batch_first=False,
                    projection_dim=hidden_dim * self.kernel_num[-1],
                )
            )
            self.enhance1 = nn.Sequential(*rnn1)
            self.enhance2 = nn.Sequential(*rnn2)
        else:
            self.enhance = nn.LSTM(
                input_size=hidden_dim * self.kernel_num[-1],
                hidden_size=self.rnn_units,
                num_layers=2,
                dropout=0.0,
                bidirectional=bidirectional,
                batch_first=False
            )
            self.tranform = nn.Linear(self.rnn_units * fac, hidden_dim * self.kernel_num[-1])

        if cfg.skip_type:
            for idx in range(len(self.kernel_num) - 1, 0, -1):
                if idx != 1:
                    self.decoder.append(
                        nn.Sequential(
                            ComplexConvTranspose2d(
                                self.kernel_num[idx] * 2,
                                self.kernel_num[idx - 1],
                                kernel_size=(self.kernel_size, 2),
                                stride=(2, 1),
                                padding=(2, 0),
                                output_padding=(1, 0)
                            ),
                            nn.BatchNorm2d(self.kernel_num[idx - 1]) if not use_cbn else ComplexBatchNorm(
                                self.kernel_num[idx - 1]),
                            # nn.ELU()
                            nn.PReLU()
                        )
                    )
                else:
                    self.decoder.append(
                        nn.Sequential(
                            ComplexConvTranspose2d(
                                self.kernel_num[idx] * 2,
                                self.kernel_num[idx - 1],
                                kernel_size=(self.kernel_size, 2),
                                stride=(2, 1),
                                padding=(2, 0),
                                output_padding=(1, 0)
                            ),
                        )
                    )
        else:
            for idx in range(len(self.kernel_num) - 1, 0, -1):
                if idx != 1:
                    self.decoder.append(
                        nn.Sequential(
                            ComplexConvTranspose2d(
                                self.kernel_num[idx],
                                self.kernel_num[idx - 1],
                                kernel_size=(self.kernel_size, 2),
                                stride=(2, 1),
                                padding=(2, 0),
                                output_padding=(1, 0)
                            ),
                            nn.BatchNorm2d(self.kernel_num[idx - 1]) if not use_cbn else ComplexBatchNorm(
                                self.kernel_num[idx - 1]),
                            # nn.ELU()
                            nn.PReLU()
                        )
                    )
                else:
                    self.decoder.append(
                        nn.Sequential(
                            ComplexConvTranspose2d(
                                self.kernel_num[idx],
                                self.kernel_num[idx - 1],
                                kernel_size=(self.kernel_size, 2),
                                stride=(2, 1),
                                padding=(2, 0),
                                output_padding=(1, 0)
                            ),
                        )
                    )
        self.flatten_parameters()

    def flatten_parameters(self):
        if isinstance(self.enhance1, nn.LSTM):
            self.enhance1.flatten_parameters()
        if isinstance(self.enhance2, nn.LSTM):
            self.enhance2.flatten_parameters()

    def forward(self, targets, inputs, direct_mapping=False):
        ###############################################################################
        #                                  for target                                 #
        ###############################################################################
        targets_specs = self.stft(targets)
        targets_real = targets_specs[:, :self.fft_len // 2 + 1]
        targets_imag = targets_specs[:, self.fft_len // 2 + 1:]
        targets_cspecs = torch.stack([targets_real, targets_imag], 1)
        targets_cspecs = targets_cspecs[:, :, 1:]

        target_encoder_out = []
        for idx, layer in enumerate(self.encoder):
            targets_cspecs = layer(targets_cspecs)
            target_encoder_out.append(targets_cspecs)

        batch_size, channels, dims, lengths = targets_cspecs.size()
        targets_mid = targets_cspecs.permute(3, 0, 1, 2)

        if cfg.lstm == 'complex':
            targets_r_rnn_in = targets_mid[:, :, :channels // 2]
            targets_i_rnn_in = targets_mid[:, :, channels // 2:]
            targets_r_rnn_in = torch.reshape(targets_r_rnn_in, [lengths, batch_size, channels // 2 * dims])
            targets_i_rnn_in = torch.reshape(targets_i_rnn_in, [lengths, batch_size, channels // 2 * dims])

            targets_r_rnn_in, targets_i_rnn_in = self.enhance1([targets_r_rnn_in, targets_i_rnn_in])
            targets_mid = torch.cat([targets_r_rnn_in, targets_i_rnn_in], 2)
            targets_mid = targets_mid.permute(1, 2, 0)

        else:
            # to [L, B, C, D]
            targets_mid = torch.reshape(targets_mid, [lengths, batch_size, channels * dims])
            targets_mid, _ = self.enhance(targets_mid)
            targets_mid = self.tranform(targets_mid)
            targets_mid = torch.reshape(targets_mid, [lengths, batch_size, channels, dims])

        # targets_mid = targets_mid.permute(1, 2, 3, 0)
        # target_out = targets_mid
        #
        # if cfg.skip_type:  # use skip connection
        #     for idx in range(len(self.decoder)):
        #         target_out = complex_cat([target_out, target_encoder_out[-1 - idx]], 1)
        #         target_out = self.decoder[idx](target_out)
        #         target_out = target_out[..., 1:]  #
        # else:
        #     for idx in range(len(self.decoder)):
        #         target_out = self.decoder[idx](target_out)
        #         target_out = target_out[..., 1:]
        #
        # if direct_mapping:  # for direct mapping model or Cyclic model
        #     target_out_real = target_out[:, 0]
        #     target_out_imag = target_out[:, 1]
        #
        #     target_out_spec = torch.cat([target_out_real, target_out_imag], 1)
        #     target_out_spec = F.pad(target_out_spec, [0, 0, 2, 0])

        ###############################################################################
        #                                  for input                                 #
        ###############################################################################
        specs = self.stft(inputs)
        real = specs[:, :self.fft_len // 2 + 1]
        imag = specs[:, self.fft_len // 2 + 1:]
        spec_mags = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)

        spec_phase = torch.atan2(imag, real)
        spec_phase = spec_phase
        cspecs = torch.stack([real, imag], 1)
        cspecs = cspecs[:, :, 1:]
        '''
        means = torch.mean(cspecs, [1,2,3], keepdim=True)
        std = torch.std(cspecs, [1,2,3], keepdim=True )
        normed_cspecs = (cspecs-means)/(std+1e-8)
        out = normed_cspecs
        '''

        out = cspecs
        encoder_out = []

        for idx, layer in enumerate(self.encoder):
            out = layer(out)
            #    print('encoder', out.size())
            encoder_out.append(out)

        batch_size, channels, dims, lengths = out.size()
        out = out.permute(3, 0, 1, 2)
        if cfg.lstm == 'complex':
            r_rnn_in = out[:, :, :channels // 2]
            i_rnn_in = out[:, :, channels // 2:]
            r_rnn_in = torch.reshape(r_rnn_in, [lengths, batch_size, channels // 2 * dims])
            i_rnn_in = torch.reshape(i_rnn_in, [lengths, batch_size, channels // 2 * dims])

            r_rnn_in, i_rnn_in = self.enhance1([r_rnn_in, i_rnn_in])

            ## for mid value (latent vector)
            output_mid = torch.cat([r_rnn_in, i_rnn_in], 2)
            output_mid = output_mid.permute(1, 2, 0)
            ##
            r_rnn_in, i_rnn_in = self.enhance2([r_rnn_in, i_rnn_in])

            r_rnn_in = torch.reshape(r_rnn_in, [lengths, batch_size, channels // 2, dims])
            i_rnn_in = torch.reshape(i_rnn_in, [lengths, batch_size, channels // 2, dims])
            out = torch.cat([r_rnn_in, i_rnn_in], 2)

        else:
            # to [L, B, C, D]
            out = torch.reshape(out, [lengths, batch_size, channels * dims])
            out, _ = self.enhance(out)
            out = self.tranform(out)
            out = torch.reshape(out, [lengths, batch_size, channels, dims])

        out = out.permute(1, 2, 3, 0)

        if cfg.skip_type:  # use skip connection
            for idx in range(len(self.decoder)):
                out = complex_cat([out, encoder_out[-1 - idx]], 1)
                out = self.decoder[idx](out)
                out = out[..., 1:]  #
        else:
            for idx in range(len(self.decoder)):
                out = self.decoder[idx](out)
                out = out[..., 1:]

        if direct_mapping:  # for direct mapping model or Cyclic model
            out_real = out[:, 0]
            out_imag = out[:, 1]

            out_spec = torch.cat([out_real, out_imag], 1)
            out_spec = F.pad(out_spec, [0, 0, 2, 0])
        else:
            #    print('decoder', out.size())
            mask_real = out[:, 0]
            mask_imag = out[:, 1]
            mask_real = F.pad(mask_real, [0, 0, 1, 0])
            mask_imag = F.pad(mask_imag, [0, 0, 1, 0])

            if self.masking_mode == 'E':
                mask_mags = (mask_real ** 2 + mask_imag ** 2) ** 0.5
                real_phase = mask_real / (mask_mags + 1e-8)
                imag_phase = mask_imag / (mask_mags + 1e-8)
                mask_phase = torch.atan2(
                    imag_phase,
                    real_phase
                )

                # mask_mags = torch.clamp_(mask_mags,0,100)
                mask_mags = torch.tanh(mask_mags)
                est_mags = mask_mags * spec_mags
                est_phase = spec_phase + mask_phase
                out_real = est_mags * torch.cos(est_phase)
                out_imag = est_mags * torch.sin(est_phase)
            elif self.masking_mode == 'C':
                out_real, out_imag = real * mask_real - imag * mask_imag, real * mask_imag + imag * mask_real
            elif self.masking_mode == 'R':
                out_real, out_imag = real * mask_real, imag * mask_imag

            out_spec = torch.cat([out_real, out_imag], 1)

        out_wav = self.istft(out_spec)
        out_wav = torch.squeeze(out_wav, 1)
        out_wav = torch.clamp_(out_wav, -1, 1)

        # # for min-max normalization
        # t = targets_mid.clone()
        # t_p_max = t.max()
        # t_n_max = t.min()
        # if abs(t_n_max) > t_p_max:
        #     targets_mid /= abs(t_n_max)
        # else:
        #     targets_mid /= t_p_max
        #
        # o = output_mid.clone()
        # o_p_max = o.max()
        # o_n_max = o.min()
        # if abs(o_n_max) > o_p_max:
        #     output_mid /= abs(o_n_max)
        # else:
        #     output_mid /= o_p_max

        # return targets_mid, output_mid, target_out_spec, out_spec, out_wav
        return targets_mid, output_mid, out_wav

    def get_params(self, weight_decay=0.0):
        # add L2 penalty
        weights, biases = [], []
        for name, param in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{
            'params': weights,
            'weight_decay': weight_decay,
        }, {
            'params': biases,
            'weight_decay': 0.0,
        }]
        return params

    def loss(self, estimated, target, real_spec=0, img_spec=0, perceptual=False):
        if perceptual:
            if cfg.perceptual == 'LMS':
                # for lms loss calculation
                clean_specs = self.stft(target)
                clean_real = clean_specs[:, :self.fft_len // 2 + 1]
                clean_imag = clean_specs[:, self.fft_len // 2 + 1:]
                clean_mags = torch.sqrt(clean_real ** 2 + clean_imag ** 2 + 1e-7)

                est_clean_mags = torch.sqrt(real_spec ** 2 + img_spec ** 2 + 1e-7)
                lms_loss = get_array_lms_loss(clean_mags, est_clean_mags)
                return lms_loss
            elif cfg.perceptual == 'PMSQE':
                ref_wav = target.reshape(-1, 3, 16000)  # dataset data shape
                est_wav = estimated.reshape(-1, 3, 16000)
                ref_wav = ref_wav.cpu()
                est_wav = est_wav.cpu()

                ref_spec = transforms.take_mag(pmsqe_stft(ref_wav))
                est_spec = transforms.take_mag(pmsqe_stft(est_wav))

                loss = pmsqe(ref_spec, est_spec)
                loss = loss.cuda()
                return loss
        else:
            if cfg.loss == 'MSE':
                return F.mse_loss(estimated, target, reduction='mean')
            elif cfg.loss == 'SDR':
                return -sdr(target, estimated)
            elif cfg.loss == 'SI-SNR':
                return -(si_snr(estimated, target))
            elif cfg.loss == 'SI-SDR':
                return -(si_sdr(target, estimated))


class complex_conditional_model(nn.Module):

    def __init__(
            self,
            rnn_layers=cfg.rnn_layers,
            rnn_units=cfg.rnn_units,
            win_len=cfg.win_len,
            win_inc=cfg.win_inc,
            fft_len=cfg.fft_len,
            win_type=cfg.window,
            masking_mode=None if cfg.masking_mode == 'Direct(None make)' else cfg.masking_mode,
            use_cbn=True if cfg.batch_norm == 'complex' else False,
            kernel_size=5
    ):
        '''
            rnn_layers: the number of lstm layers in the crn,
            rnn_units: for clstm, rnn_units = real+imag
        '''

        super(complex_conditional_model, self).__init__()

        # for fft
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.win_type = win_type

        input_dim = win_len
        output_dim = win_len

        self.rnn_units = rnn_units
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = rnn_layers
        self.kernel_size = kernel_size
        kernel_num = cfg.dccrn_kernel_num
        self.kernel_num = [2] + kernel_num
        self.masking_mode = masking_mode

        # bidirectional=True
        bidirectional = False
        fac = 2 if bidirectional else 1

        fix = True
        self.fix = fix
        self.stft = ConvSTFT(self.win_len, self.win_inc, fft_len, self.win_type, 'complex', fix=fix)
        self.istft = ConviSTFT(self.win_len, self.win_inc, fft_len, self.win_type, 'complex', fix=fix)

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for idx in range(len(self.kernel_num) - 1):
            self.encoder.append(
                nn.Sequential(
                    # nn.ConstantPad2d([0, 0, 0, 0], 0),
                    ComplexConv2d(
                        self.kernel_num[idx],
                        self.kernel_num[idx + 1],
                        kernel_size=(self.kernel_size, 2),
                        stride=(2, 1),
                        padding=(2, 1)
                    ),
                    nn.BatchNorm2d(self.kernel_num[idx + 1]) if not use_cbn else ComplexBatchNorm(
                        self.kernel_num[idx + 1]),
                    nn.PReLU()
                )
            )
        hidden_dim = self.fft_len // (2 ** (len(self.kernel_num)))

        if cfg.lstm == 'complex':
            rnns = []
            for idx in range(rnn_layers):
                rnns.append(
                    NavieComplexLSTM(
                        input_size=hidden_dim * self.kernel_num[-1] if idx == 0 else self.rnn_units,
                        hidden_size=self.rnn_units,
                        bidirectional=bidirectional,
                        batch_first=False,
                        projection_dim=hidden_dim * self.kernel_num[-1] if idx == rnn_layers - 1 else None,
                    )
                )
                self.enhance = nn.Sequential(*rnns)
        else:
            self.enhance = nn.LSTM(
                input_size=hidden_dim * self.kernel_num[-1],
                hidden_size=self.rnn_units,
                num_layers=2,
                dropout=0.0,
                bidirectional=bidirectional,
                batch_first=False
            )
            self.tranform = nn.Linear(self.rnn_units * fac, hidden_dim * self.kernel_num[-1])

        if cfg.skip_type:
            for idx in range(len(self.kernel_num) - 1, 0, -1):
                if idx != 1:
                    self.decoder.append(
                        nn.Sequential(
                            ComplexConvTranspose2d(
                                self.kernel_num[idx] * 2,
                                self.kernel_num[idx - 1],
                                kernel_size=(self.kernel_size, 2),
                                stride=(2, 1),
                                padding=(2, 0),
                                output_padding=(1, 0)
                            ),
                            nn.BatchNorm2d(self.kernel_num[idx - 1]) if not use_cbn else ComplexBatchNorm(
                                self.kernel_num[idx - 1]),
                            # nn.ELU()
                            nn.PReLU()
                        )
                    )
                else:
                    self.decoder.append(
                        nn.Sequential(
                            ComplexConvTranspose2d(
                                self.kernel_num[idx] * 2,
                                self.kernel_num[idx - 1],
                                kernel_size=(self.kernel_size, 2),
                                stride=(2, 1),
                                padding=(2, 0),
                                output_padding=(1, 0)
                            ),
                        )
                    )
        else:
            for idx in range(len(self.kernel_num) - 1, 0, -1):
                if idx != 1:
                    self.decoder.append(
                        nn.Sequential(
                            ComplexConvTranspose2d(
                                self.kernel_num[idx],
                                self.kernel_num[idx - 1],
                                kernel_size=(self.kernel_size, 2),
                                stride=(2, 1),
                                padding=(2, 0),
                                output_padding=(1, 0)
                            ),
                            nn.BatchNorm2d(self.kernel_num[idx - 1]) if not use_cbn else ComplexBatchNorm(
                                self.kernel_num[idx - 1]),
                            # nn.ELU()
                            nn.PReLU()
                        )
                    )
                else:
                    self.decoder.append(
                        nn.Sequential(
                            ComplexConvTranspose2d(
                                self.kernel_num[idx],
                                self.kernel_num[idx - 1],
                                kernel_size=(self.kernel_size, 2),
                                stride=(2, 1),
                                padding=(2, 0),
                                output_padding=(1, 0)
                            ),
                        )
                    )
        self.flatten_parameters()

    def flatten_parameters(self):
        if isinstance(self.enhance, nn.LSTM):
            self.enhance.flatten_parameters()

    def forward(self, conditions, inputs, direct_mapping=False):
        specs = self.stft(inputs)
        real = specs[:, :self.fft_len // 2 + 1]
        imag = specs[:, self.fft_len // 2 + 1:]
        spec_mags = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)

        spec_phase = torch.atan2(imag, real)
        spec_phase = spec_phase
        cspecs = torch.stack([real, imag], 1)
        cspecs = cspecs[:, :, 1:]
        '''
        means = torch.mean(cspecs, [1,2,3], keepdim=True)
        std = torch.std(cspecs, [1,2,3], keepdim=True )
        normed_cspecs = (cspecs-means)/(std+1e-8)
        out = normed_cspecs
        '''

        out = cspecs
        encoder_out = []

        for idx, layer in enumerate(self.encoder):
            out = layer(out)
            #    print('encoder', out.size())
            encoder_out.append(out)

        batch_size, channels, dims, lengths = out.size()
        out = out.permute(3, 0, 1, 2)
        if cfg.lstm == 'complex':
            r_rnn_in = out[:, :, :channels // 2]
            i_rnn_in = out[:, :, channels // 2:]
            r_rnn_in = torch.reshape(r_rnn_in, [lengths, batch_size, channels // 2 * dims])
            i_rnn_in = torch.reshape(i_rnn_in, [lengths, batch_size, channels // 2 * dims])

            r_rnn_in, i_rnn_in = self.enhance([r_rnn_in, i_rnn_in])

            r_rnn_in = torch.reshape(r_rnn_in, [lengths, batch_size, channels // 2, dims])
            i_rnn_in = torch.reshape(i_rnn_in, [lengths, batch_size, channels // 2, dims])
            out = torch.cat([r_rnn_in, i_rnn_in], 2)

        else:
            # to [L, B, C, D]
            out = torch.reshape(out, [lengths, batch_size, channels * dims])
            out, _ = self.enhance(out)
            out = self.tranform(out)
            out = torch.reshape(out, [lengths, batch_size, channels, dims])

        out = out.permute(1, 2, 3, 0)

        if cfg.skip_type:  # use skip connection
            for idx in range(len(self.decoder)):
                out = complex_cat([out, encoder_out[-1 - idx]], 1)
                out = self.decoder[idx](out)
                out = out[..., 1:]  #
        else:
            for idx in range(len(self.decoder)):
                out = self.decoder[idx](out)
                out = out[..., 1:]

        if direct_mapping:  # for direct mapping model or Cyclic model
            out_real = out[:, 0]
            out_imag = out[:, 1]

            out_spec = torch.cat([out_real, out_imag], 1)
            out_spec = F.pad(out_spec, [0, 0, 2, 0])
        else:
            #    print('decoder', out.size())
            mask_real = out[:, 0]
            mask_imag = out[:, 1]
            mask_real = F.pad(mask_real, [0, 0, 1, 0])
            mask_imag = F.pad(mask_imag, [0, 0, 1, 0])

            if self.masking_mode == 'E':
                mask_mags = (mask_real ** 2 + mask_imag ** 2) ** 0.5
                real_phase = mask_real / (mask_mags + 1e-8)
                imag_phase = mask_imag / (mask_mags + 1e-8)
                mask_phase = torch.atan2(
                    imag_phase,
                    real_phase
                )

                # mask_mags = torch.clamp_(mask_mags,0,100)
                mask_mags = torch.tanh(mask_mags)
                est_mags = mask_mags * spec_mags
                est_phase = spec_phase + mask_phase
                out_real = est_mags * torch.cos(est_phase)
                out_imag = est_mags * torch.sin(est_phase)
            elif self.masking_mode == 'C':
                out_real, out_imag = real * mask_real - imag * mask_imag, real * mask_imag + imag * mask_real
            elif self.masking_mode == 'R':
                out_real, out_imag = real * mask_real, imag * mask_imag

            out_spec = torch.cat([out_real, out_imag], 1)

        out_wav = self.istft(out_spec)
        out_wav = torch.squeeze(out_wav, 1)
        out_wav = torch.clamp_(out_wav, -1, 1)

        return out_real, out_imag, out_wav

    def get_params(self, weight_decay=0.0):
        # add L2 penalty
        weights, biases = [], []
        for name, param in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{
            'params': weights,
            'weight_decay': weight_decay,
        }, {
            'params': biases,
            'weight_decay': 0.0,
        }]
        return params

    def loss(self, estimated, target, real_spec=0, img_spec=0, GMT=0, perceptual=False):
        if perceptual:
            if cfg.perceptual == 'LMS':
                # for lms loss calculation
                clean_specs = self.stft(target)
                clean_real = clean_specs[:, :self.fft_len // 2 + 1]
                clean_imag = clean_specs[:, self.fft_len // 2 + 1:]
                clean_mags = torch.sqrt(clean_real ** 2 + clean_imag ** 2 + 1e-7)

                est_clean_mags = torch.sqrt(real_spec ** 2 + img_spec ** 2 + 1e-7)
                lms_loss = get_array_lms_loss(clean_mags, est_clean_mags)
                return lms_loss
            elif cfg.perceptual == 'PMSQE':
                ref_wav = target.reshape(-1, 3, 16000)  # dataset data shape
                est_wav = estimated.reshape(-1, 3, 16000)
                ref_wav = ref_wav.cpu()
                est_wav = est_wav.cpu()

                ref_spec = transforms.take_mag(pmsqe_stft(ref_wav))
                est_spec = transforms.take_mag(pmsqe_stft(est_wav))

                loss = pmsqe(ref_spec, est_spec)
                loss = loss.cuda()
                return loss
            elif cfg.perceptual == 'PAM':
                residual_noise = estimated - target
                noise_spec = self.stft(residual_noise)
                noise_real = noise_spec[:, : self.fft_len // 2 + 1]
                noise_img = noise_spec[:, self.fft_len // 2 + 1:]

                noise_mag = torch.sqrt(noise_real ** 2 + noise_img ** 2 + 1e-7)

                pam_loss = get_pam_loss(noise_mag, GMT)
                return pam_loss
        else:
            if cfg.loss == 'MSE':
                return F.mse_loss(estimated, target, reduction='mean')
            elif cfg.loss == 'SDR':
                return -sdr(target, estimated)
            elif cfg.loss == 'SI-SNR':
                return -(si_snr(estimated, target))
            elif cfg.loss == 'SI-SDR':
                return -(si_sdr(target, estimated))
