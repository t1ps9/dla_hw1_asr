import re
from string import ascii_lowercase
from collections import defaultdict

import torch
from pyctcdecode import Alphabet, BeamSearchDecoderCTC

# TODO add CTC decode
# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        self.beam_search_decode = BeamSearchDecoderCTC(Alphabet(self.vocab, False), None)

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        decoded = []
        last_ind = self.char2ind[self.EMPTY_TOK]
        for ind in inds:
            if ind == last_ind:
                continue
            elif ind != self.char2ind[self.EMPTY_TOK]:
                decoded.append(self.ind2char[ind])
            last_ind = ind
        return "".join(decoded)

    # library
    # def ctc_beam_search(self, log_probs, beam_size=3) -> str:
    #     log_probs = log_probs.cpu().numpy()
    #     return self.beam_search_decode.decode(log_probs, beam_size)

    # implementation

    def ctc_beam_search(self, probs, beam_size=3):
        probs = torch.exp(probs)
        dp = {
            ("", self.EMPTY_TOK): 1.0,
        }
        for prob in probs:
            dp = self._expand_and_merge_path(dp, prob)
            dp = self._truncate_paths(dp, beam_size)
        dp = [
            (prefix, proba)
            for (prefix, _), proba in sorted(dp.items(), key=lambda x: -x[1])
        ]
        return dp[0][0]

    def _expand_and_merge_path(self, dp, next_token_probs):
        new_dp = defaultdict(float)
        for ind, next_token_prob in enumerate(next_token_probs):
            cur_char = self.ind2char[ind]
            for (prefix, last_char), v in dp.items():
                if last_char == cur_char:
                    new_prefix = prefix
                else:
                    if cur_char != self.EMPTY_TOK:
                        new_prefix = prefix + cur_char
                    else:
                        new_prefix = prefix
                new_dp[(new_prefix, cur_char)] += v * next_token_prob
        return new_dp

    def _truncate_paths(self, dp, beam_size=3):
        return dict(sorted(list(dp.items()), key=lambda x: -x[1])[:beam_size])

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
