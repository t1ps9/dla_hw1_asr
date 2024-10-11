import torch
import torch.nn as nn
import torchaudio


class ApplyFrequencyMasking(nn.Module):
    def __init__(self, freq_mask_param, probability=0.3):
        super(ApplyFrequencyMasking, self).__init__()
        self.freq_mask_param = freq_mask_param
        self.probability = probability
        self.masking_fn = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)

    def forward(self, spectrogram):
        if torch.rand(1).item() < self.probability:
            return self.masking_fn(spectrogram)
        return spectrogram


class ApplyTimeMasking(nn.Module):
    def __init__(self, time_mask_param, probability=0.3):
        super(ApplyTimeMasking, self).__init__()
        self.time_mask_param = time_mask_param
        self.probability = probability
        self.masking_fn = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)

    def forward(self, spectrogram):
        if torch.rand(1).item() < self.probability:
            return self.masking_fn(spectrogram)
        return spectrogram
