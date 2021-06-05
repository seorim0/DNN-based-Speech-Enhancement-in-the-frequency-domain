import os
import torch
import torch.nn as nn
import numpy as np
from scipy.signal import get_window
import torch.nn.functional as F
from pathlib import Path
import config as cfg
import soundfile
from matplotlib import pyplot as plt


############################################################################
#                            Tools for PAM                                 #
############################################################################
# --------------------------------------------------------------------
# Step-1: Spectral analysis and SPL Normalization
# --------------------------------------------------------------------
def spectral_analysis_SPL_normalization(current_frame_mag):
    PN = 90.302

    x = current_frame_mag / cfg.fft_len
    P = PN + 10 * np.log10(current_frame_mag ** 2 + 1e-16)  # Only first half is required

    return P


# --------------------------------------------------------------------
# Step-2: Identification of tonal and noise maskers
# --------------------------------------------------------------------

# SUBFUNCTIONS BEGIN
def tone_masker_check(P, k):
    # If P(k) is a local maxima and is greater than 7dB in
    # a frequency dependent neighborhood, it is a tone.         See section (5.7.2)------->>>
    # This neighborhood is defined as:                                  Eq. (5.22)------>>>
    #   within 2           if 1   < k  < 62, for frequencies between 0.17-5.5kHz
    #   within 2,3            62  <= k < 126, for frequencies between 5.5-11Khz
    #   within 2,3,4,5,6      126 <= k < 255, for frequencies between 11-20Khz

    # If it is at the beginning or end of P, then it is not a local maxima
    # The if... else... statements below computes the tonal set given by Eq. (5.21) ------>>>
    if (k <= 0 or k >= 249):
        bool_value = 0

    # if it's not a local maxima, leave with bool=0
    elif ((P[k] < P[k - 1]) or (P[k] < P[k + 1])):
        bool_value = 0

    # otherwise, we need to check if it is a max in its neighborhood.
    elif ((k > 1) and (k < 62)):
        bool_value = (P[k] > (P[k - 2] + 7)) and (P[k] > (P[k + 2] + 7))
    elif ((k >= 62) and (k < 126)):
        bool_value = (P[k] > (P[k - 2] + 7)) and (P[k] > (P[k + 2] + 7)) and (P[k] > (P[k - 3] + 7)) and (
                P[k] > (P[k + 3] + 7))
    elif ((k >= 126) and (k <= 255)):
        bool_value = (P[k] > (P[k - 2] + 7)) and (P[k] > (P[k + 2] + 7)) and (P[k] > (P[k - 3] + 7)) and (
                P[k] > (P[k + 3] + 7)) \
                     and (P[k] > (P[k - 4] + 7)) and (P[k] > (P[k + 4] + 7)) and (P[k] > (P[k - 5] + 7)) and (
                             P[k] > (P[k + 5] + 7)) \
                     and (P[k] > (P[k - 6] + 7)) and (P[k] > (P[k + 6] + 7))
    else:
        bool_value = 0
    return bool_value


def noise_masker_check(psd, tone_masker, low, high):
    noise_members = np.ones(high - low)

    # Browse through the power spectral density, P to determine the noise maskers
    for k in range(low, high, 1):
        # if there is a tone
        if (tone_masker[k] > 0):
            # check frequency location and determine neighborhood length
            if ((k > 1) and (k < 62)):
                m = 2
            elif ((k >= 62) and (k < 126)):
                m = 3
            elif ((k >= 126) and (k < 255)):
                m = 6
            else:
                m = 0

            # set all members of the neighborhood to 0 that removes them from the list of noise members
            if noise_members.shape[0] < (k - low) + m + 1:
                noise_members = np.concatenate([noise_members, np.zeros((k - low) + m + 1 - len(noise_members))])
            for n in range((k - low) - m, (k - low) + m + 1):
                if (n >= 0):
                    noise_members[n] = 0

    # if there are no noise members in the range, then leave
    if np.where(noise_members == 1)[0].size == 0:
        noise_masker_at_loc = 0
        loc = -1
    else:
        temp = 0
        for k in low + np.where(noise_members == 1)[0]:
            temp += 10 ** (0.1 * psd[k])
        noise_masker_at_loc = 10 * np.log10(temp)
        # geomean
        loc = np.exp(
            np.sum(np.log(low + np.where(noise_members == 1)[0] + 1)) / len(low + np.where(noise_members == 1)[0] + 1))

    return noise_masker_at_loc, loc


def Identification_tonal_noise_maskers(P, freq_bark):
    # Browse through the power spectral density, P
    # to determine the tone maskers --------------------------------->>>>>>> Part-2(a)
    P_TM = np.zeros(len(P))

    for k in range(len(P)):
        if tone_masker_check(P, k):
            # if index k corresponds to a tone
            # Combine the energy from three adjacent spectral components
            # centered at the peak to form a single tonal masker
            P_TM[k] = 10 * np.log10(10 ** (0.1 * P[k - 1]) + 10 ** (0.1 * P[k]) + 10 ** (0.1 * P[k + 1]))  # Eq. (5.23)

    # Find noise maskers within the critical band-------------------------------->>>>>>> Part-2(b)
    P_NM = np.zeros(len(P_TM))
    lowbin = 0  # lower spectral line boundary of the critical bank, l
    highbin = np.max(np.where(freq_bark < 1)) + 1  # upper spectral line boundary of the critical bank, u
    # loc is the geometric mean spectral line of the critical band, Eq. (5.25)
    for band in range(24):
        noise_masker_at_loc, loc = noise_masker_check(P, P_TM, lowbin, highbin)
        if (loc != -1):
            P_NM[int(np.floor(loc)) - 1] = noise_masker_at_loc
        lowbin = highbin - 1
        highbin = np.max(np.where(freq_bark < (band + 2))) + 1

    return P_TM, P_NM


# --------------------------------------------------------------------
# Step-3: Decimation and re-organization of maskers
# --------------------------------------------------------------------
def Decimation(tone_masker, noise_masker, freq_bark, Abs_thr):
    TM_above_thres = tone_masker * (tone_masker > Abs_thr)
    NM_above_thres = noise_masker * (noise_masker > Abs_thr)

    # The remaining maskers must now be checked to see if any are
    # within a critical band.  If they are, then only the strongest
    # one matters.  The other can be set to zero.
    # go through masker list
    for j in range(len(Abs_thr)):
        toneFound = 0
        noiseFound = 0
        # was a tone or noise masker found?
        if (TM_above_thres[j] > 0):
            toneFound = 1

        # if either masker found
        if (toneFound or noiseFound):
            masker_loc_barks = freq_bark[j]
            # determine low and high thresholds of critical band
            crit_bw_low = masker_loc_barks - 0.5
            crit_bw_high = masker_loc_barks + 0.5
            # determine what indices these values correspond to

            if (np.where(freq_bark < crit_bw_low)[0]).size != 0:
                low_loc = np.max(np.where(freq_bark < crit_bw_low))
                low_loc += 1
            else:
                low_loc = 0

            if (np.where(freq_bark < crit_bw_high)[0]).size != 0:
                high_loc = np.max(np.where(freq_bark < crit_bw_high)) + 1

            # At this point, we know the location of a masker and its
            # critical band.  Depending on which type of masker it is,
            # browse through and eliminate the maskers within the critical band that are lower.
            for k in range(low_loc, high_loc):
                if (toneFound):
                    # find other tone maskers in critical band
                    if ((TM_above_thres[j] < TM_above_thres[k]) and (k != j)):
                        TM_above_thres[j] = 0
                        break
                    elif (k != j):
                        TM_above_thres[k] = 0

                    # find noise maskers in critical band
                    if (TM_above_thres[j] < NM_above_thres[k]):
                        TM_above_thres[j] = 0
                        break
                    else:
                        NM_above_thres[k] = 0

                elif (noiseFound):
                    # find other noise maskers in critical band
                    if ((NM_above_thres[j] < NM_above_thres[k]) and (k != j)):
                        NM_above_thres[j] = 0
                        break
                    elif (k != j):
                        NM_above_thres[k] = 0

                    # find tone maskers in critical band
                    if (NM_above_thres[j] < TM_above_thres[k]):
                        NM_above_thres[j] = 0
                        break
                    else:
                        TM_above_thres[k] = 0

    return TM_above_thres, NM_above_thres


# --------------------------------------------------------------------
# Step-4: Calculation of individual masking thresholds
# --------------------------------------------------------------------

# SUBFUNCTIONS BEGIN
def spreading_function(masker_bin, power, low, high, bark):  ### Eq. (5.31) ---->>>>
    spread = np.zeros(high - low)
    masker_bark = bark[masker_bin]
    for i in range(low, high, 1):
        maskee_bark = bark[i]
        deltaz = maskee_bark - masker_bark
        if ((deltaz >= -3.5) and (deltaz < -1)):
            spread[i - low] = 17 * deltaz - 0.4 * power + 11
        elif ((deltaz >= -1) and (deltaz < 0)):
            spread[i - low] = (0.4 * power + 6) * deltaz
        elif ((deltaz >= 0) and (deltaz < 1)):
            spread[i - low] = -17 * deltaz
        elif ((deltaz >= 1) and (deltaz < 8.5)):
            spread[i - low] = (0.15 * power - 17) * deltaz - 0.15 * power

    return spread


def mask_threshold(type_thr, j, P, bark):
    # mask_threshold returns an array of the masked threshold in dB SPL that
    # results around a mask located at a frequency bin (i.e., in
    # discrete terms).  It also returns a starting index for this threshold,
    # which is discussed later.

    # The user should also supply the power spectral density and the related Bark
    # spectrum so that all calculations can be made. Note also that two
    # different threshold are possible, so the user should specify:
    #
    #  type = 0      threshold = NOISE threshold
    #  type = 1      threshold = TONE  threshold

    # This thresholding is determined in a range from -3 to +8 Barks
    # from the mask. (This is why a bark spectrum is needed.)  In case you
    # would like to overlay different thresholds, you need to know where each
    # one actually starts.  Thus, the starting bin for the threshold is also
    # returned.

    # determine where masker is in barks
    maskerloc = bark[j]

    # set up range of the resulting function in barks
    low = maskerloc - 3
    high = maskerloc + 8

    # in discrete bins
    if (np.where(bark < low)[0]).size != 0:
        lowbin = np.max(np.where(bark < low))

    else:
        lowbin = 0

    if (np.where(bark < high)[0]).size != 0:
        highbin = np.max(np.where(bark < high)) + 1

    # calculate spreading function
    SF = spreading_function(j, P, lowbin, highbin, bark)

    if type_thr == 0:
        # calculate noise threshold
        threshold = P - 0.175 * bark[j] + SF - 2.025
    else:
        # calculate tone threshold
        threshold = P - 0.275 * bark[j] + SF - 6.025

    # finally, note that the lowest value in threshold corresponds to the frequency bin at lowbin
    start = lowbin

    return threshold, start


def Individual_masking_threshold(P_TM_th, P_NM_th, freq_bark):
    Thr_TM = np.zeros(len(P_TM_th))

    # Go through the tone list
    if np.where(P_TM_th != 0)[0].size != 0:
        for k in np.where(P_TM_th != 0)[0]:
            # determine the masking threshold around the tone masker
            # CALL function 5: mask_threshold
            [thres, start] = mask_threshold(1, k, P_TM_th[k], freq_bark)  ##Eq. (5.30) ---->>>
            # add the power of the threshold to temp in the proper frequency range
            Thr_TM[start:start + len(thres)] = Thr_TM[start:start + len(thres)] + 10 ** (0.1 * thres)

    Thr_NM = np.zeros(len(P_NM_th))
    if np.where(P_NM_th != 0)[0].size != 0:
        for k in np.where(P_NM_th != 0)[0]:
            # determine the masking threshold around the noise masker
            # CALL function 5: mask_threshold
            [thres, start] = mask_threshold(1, k, P_NM_th[k], freq_bark)  ##Eq. (5.32) ---->>>
            # add the power of the threshold to temp in the proper frequency range
            Thr_NM[start:start + len(thres)] = Thr_NM[start:start + len(thres)] + 10 ** (0.1 * thres)
    return Thr_TM, Thr_NM


# --------------------------------------------------------------------
# Step-5: Calculation of global masking thresholds
# --------------------------------------------------------------------
# Global_threshold takes the absolute threshold of hearing as well as the
# spectral densities of noise and tones to determine the overall global
# masking threshold.  This method assumes that the effects of masking
# are additive, so the masks of all maskers and the absolute threshold
# are added together.
def Global_masking_threshold(Thr_TM, Thr_NM, Abs_thr):
    temp = Thr_TM + Thr_NM
    # finally, add the power of the absolute hearing threshold to the list
    for k in range(len(Abs_thr)):
        # -3dB
        abs_down = Abs_thr[k] - 3

        temp[k] = temp[k] + 10 ** (0.1 * abs_down)  # Eq. (5.33)

    Thr_global = temp  # Note this is not in dB
    return Thr_global


# ----------------------------------------------------------------------
# Calculate Perceptual Entropy
# ----------------------------------------------------------------------
def calculate_PE(current_frame, Thr_global):
    x = current_frame / 512
    p = np.fft.rfft(x)

    PE_vec = np.zeros(257)
    for i in range(257):
        # 257 (linear PE)
        PE_vec[i] = (np.log2(2 * np.abs(np.sqrt(10 ** 9.0302) * np.real(p[i]) / np.sqrt(6 * Thr_global[i])) + 1) \
                     + np.log2(2 * np.abs(np.sqrt(10 ** 9.0302) * np.imag(p[i]) / np.sqrt(6 * Thr_global[i])) + 1))

    return PE_vec


def PAM_1(current_frame_mag, fs):
    freq_hz = np.arange(1, int(cfg.fft_len / 2) + 2) * (fs / cfg.fft_len)  # Freq. bins in Hz
    freq_bark = 13 * np.arctan(.00076 * freq_hz) + 3.5 * np.arctan(
        (freq_hz / 7500) ** 2)  # Eq. (5.3)   Bark indices corresponding to freq. bins
    Abs_thr = 3.64 * (freq_hz / 1000) ** (-.8) - 6.5 * np.exp(-0.6 * (freq_hz / 1000 - 3.3) ** 2) + 0.001 * (
            freq_hz / 1000) ** 4  # Eq. (5.1)   Absolute Threshold in quiet

    # Step-1: Spectral analysis and SPL Normalization
    P = spectral_analysis_SPL_normalization(current_frame_mag)

    # Step-2: Identification of tonal and noise maskers
    P_TM, P_NM = Identification_tonal_noise_maskers(P, freq_bark)

    # Step-3: Decimation and re-organization of maskers
    P_TM_th, P_NM_th = Decimation(P_TM, P_NM, freq_bark, Abs_thr)

    # Step-4: Calculation of individual masking thresholds
    Thr_TM, Thr_NM = Individual_masking_threshold(P_TM_th, P_NM_th, freq_bark)

    # Step-5: Calculation of global masking thresholds
    Thr_global = Global_masking_threshold(Thr_TM, Thr_NM, Abs_thr)

    return Thr_global


def scan_clean_only(dir_name):
    """Scan directory and save address of clean/noisy wav data.
        Args:
            dir_name: directroy name to scan
        Returns:
            addr: all address list of clean/noisy wave data in subdirectory
        """
    if os.path.isdir(dir_name) is False:
        print("[Error] There is no directory '%s'." % dir_name)
        exit()
    else:
        print("Scanning a directory %s " % dir_name)

    addr = []
    for subdir, dirs, files in os.walk(dir_name):
        for file in files:
            if file.endswith(".wav"):
                filepath = Path(subdir) / file
                addr_clean = filepath
                addr.append(addr_clean)
    return addr


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
