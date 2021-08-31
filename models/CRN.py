#######################################################################
#                            real network                             #
#######################################################################
class CRN(nn.Module):
    def __init__(
            self,
            rnn_layers=rnn_layers,
            rnn_units=rnn_units,
            win_len=win_len,
            win_inc=win_inc,
            fft_len=fft_len,
            win_type=window,
            kernel_size=5
    ):
        '''
            rnn_layers: the number of lstm layers in the crn
        '''

        super(CRN, self).__init__()

        # for fft
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.win_type = win_type

        input_dim = win_len
        output_dim = win_len

        self.rnn_input_size = rnn_input_size
        self.rnn_units = rnn_units
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = rnn_layers
        self.kernel_size = kernel_size
        kernel_num = dccrn_kernel_num
        self.kernel_num = [2] + kernel_num

        # bidirectional=True
        bidirectional = False
        fac = 2 if bidirectional else 1

        self.stft = ConvSTFT(self.win_len, self.win_inc, fft_len, self.win_type, 'real')
        self.istft = ConviSTFT(self.win_len, self.win_inc, fft_len, self.win_type, 'real')

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for idx in range(len(self.kernel_num) - 1):
            self.encoder.append(
                nn.Sequential(
                    RealConv2d(
                        self.kernel_num[idx]//2,
                        self.kernel_num[idx + 1]//2,
                        kernel_size=(self.kernel_size, 2),
                        stride=(2, 1),
                        padding=(2, 1)
                    ),
                    nn.BatchNorm2d(self.kernel_num[idx + 1]//2),
                    nn.PReLU()
                )
            )
        hidden_dim = self.fft_len // (2 ** (len(self.kernel_num)))

        self.enhance = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=self.rnn_units,
            dropout=0.0,
            bidirectional=bidirectional,
            batch_first=False
        )
        self.tranform = nn.Linear(self.rnn_units, self.rnn_input_size)

        if skip_type:
            for idx in range(len(self.kernel_num) - 1, 0, -1):
                if idx != 1:
                    self.decoder.append(
                        nn.Sequential(
                            RealConvTranspose2d(
                                self.kernel_num[idx],
                                self.kernel_num[idx - 1]//2,
                                kernel_size=(self.kernel_size, 2),
                                stride=(2, 1),
                                padding=(2, 0),
                                output_padding=(1, 0)
                            ),
                            nn.BatchNorm2d(self.kernel_num[idx - 1]//2),
                            nn.PReLU()
                        )
                    )
                else:
                    self.decoder.append(
                        nn.Sequential(
                            RealConvTranspose2d(
                                self.kernel_num[idx],
                                self.kernel_num[idx - 1]//2,
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
                            nn.ConvTranspose2d(
                                self.kernel_num[idx],
                                self.kernel_num[idx - 1],
                                kernel_size=(self.kernel_size, 2),
                                stride=(2, 1),
                                padding=(2, 0),
                                output_padding=(1, 0)
                            ),
                            nn.BatchNorm2d(self.kernel_num[idx - 1]),
                            # nn.ELU()
                            nn.PReLU()
                        )
                    )
                else:
                    self.decoder.append(
                        nn.Sequential(
                            nn.ConvTranspose2d(
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

    def forward(self, inputs, targets=0):

        mags, phase = self.stft(inputs)

        out = mags
        out = out.unsqueeze(1)
        encoder_out = []

        for idx, layer in enumerate(self.encoder):
            out = layer(out)
            #    print('encoder', out.size())
            encoder_out.append(out)

        batch_size, channels, dims, lengths = out.size()
        out = out.permute(3, 0, 1, 2)

        rnn_in = torch.reshape(out, [lengths, batch_size, channels * dims])
        out, _ = self.enhance(rnn_in)
        out = self.tranform(out)
        out = torch.reshape(out, [lengths, batch_size, channels, dims])

        out = out.permute(1, 2, 3, 0)

        if skip_type:  # use skip connection
            for idx in range(len(self.decoder)):
                out = torch.cat([out, encoder_out[-1 - idx]], 1)
                out = self.decoder[idx](out)
                out = out[..., 1:, 1:]  #
        else:
            for idx in range(len(self.decoder)):
                out = self.decoder[idx](out)
                out = out[..., 1:]

        # mask_mags = F.pad(out, [0, 0, 1, 0])
        out = out.squeeze(1)

        if direct_mapping:  # spectral mapping
            target_mags, _ = self.stft(target)

            out_real = out * torch.cos(phase)
            out_imag = out * torch.sin(phase)

            out_spec = torch.cat([out_real, out_imag], 1)

            out_wav = self.istft(out_spec)
            out_wav = torch.squeeze(out_wav, 1)
            out_wav = torch.clamp_(out_wav, -1, 1)

            return out, target_mags, out_wav
        else:  # T-F masking
            # mask_mags = torch.clamp_(mask_mags,0,100)
            mask_mags = torch.tanh(out)
            est_mags = mask_mags * mags
            out_real = est_mags * torch.cos(phase)
            out_imag = est_mags * torch.sin(phase)

            out_spec = torch.cat([out_real, out_imag], 1)

            out_wav = self.istft(out_spec)
            out_wav = torch.squeeze(out_wav, 1)
            out_wav = torch.clamp_(out_wav, -1, 1)

            return out_wav

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

    def loss(self, estimated, target):
        if current_loss == 'MSE':
            return F.mse_loss(estimated, target, reduction='mean')
        elif current_loss == 'SDR':
            return -sdr(target, estimated)
        elif current_loss == 'SI-SNR':
            return -(si_snr(estimated, target))
        elif current_loss == 'SI-SDR':
            return -(si_sdr(target, estimated))
