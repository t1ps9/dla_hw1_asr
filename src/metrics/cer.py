from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_cer

# TODO add beam search/lm versions
# Note: they can be written in a pretty way
# Note 2: overall metric design can be significantly improved


class ArgmaxCERMetric(BaseMetric):
    def __init__(self, text_encoder, decode='argmax', beam_size=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.decode = decode
        self.beam_size = beam_size

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        cers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            if self.decode == 'argmax':
                pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            elif self.decode == 'beam_search':
                pred_text = self.text_encoder.ctc_beam_search(log_prob_vec[:length], self.beam_size)
            pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)
