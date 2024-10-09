import torch
import torch.nn as nn


class DeepSpeech2(nn.Module):
    def __init__(self, n_layers, n_feats, rnn_hidden_size, p_drop, n_tokens, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_layers = n_layers
        self.n_feats = n_feats
        self.rnn_hidden_size = rnn_hidden_size
        self.p_drop = p_drop

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=96, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(),
        )

        self.n_freaq = self.n_feats
        for layer in self.convs:
            if isinstance(layer, nn.Conv2d):
                l_in = self.n_freaq
                kernel_size = layer.kernel_size[0]
                stride = layer.stride[0]
                padding = layer.padding[0]
                dilation = layer.dilation[0]
                self.n_freaq = (l_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

        rnn_input_size = self.n_freaq * 96

        self.block_rnn = nn.ModuleList()
        for i in range(self.n_layers):
            input_size = rnn_input_size if i == 0 else self.rnn_hidden_size
            rnn = nn.GRU(
                input_size=input_size,
                hidden_size=self.rnn_hidden_size,
                num_layers=1,
                batch_first=True,
                dropout=self.p_drop if i != self.n_layers - 1 else 0,
                bidirectional=True,
            )
            self.block_rnn.append(rnn)
            if i != self.n_layers - 1:
                self.block_rnn.append(nn.BatchNorm1d(self.rnn_hidden_size))

        self.fc = nn.Linear(self.rnn_hidden_size, n_tokens)

    def forward(self, spectrogram, spectrogram_length, **batch):
        x = spectrogram.unsqueeze(1)
        spectrogram_length = spectrogram_length.to(x.device)

        for layer in self.convs:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                module = layer
                length = spectrogram_length
                padding = torch.tensor(module.padding[1], device=length.device)
                dilation = torch.tensor(module.dilation[1], device=length.device)
                kernel_size = torch.tensor(module.kernel_size[1], device=length.device)
                stride = torch.tensor(module.stride[1], device=length.device)
                spectrogram_length = (
                    length
                    + 2 * padding
                    - dilation * (kernel_size - 1)
                    - 1
                ) // stride + 1

            b, c, f, t = x.shape
            time_indices = torch.arange(t, device=x.device).unsqueeze(0)
            mask = time_indices >= spectrogram_length.unsqueeze(1)
            mask = mask.unsqueeze(1).unsqueeze(2)
            x = x.masked_fill(mask, 0)

        batch_size, channels, freq, time = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(batch_size, time, -1)

        for layer in self.block_rnn:
            if isinstance(layer, nn.GRU):
                x = nn.utils.rnn.pack_padded_sequence(
                    x, spectrogram_length.cpu(), batch_first=True, enforce_sorted=False
                )
                x, _ = layer(x)
                x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
                x = x[:, :, :self.rnn_hidden_size] + x[:, :, self.rnn_hidden_size:]
            elif isinstance(layer, nn.BatchNorm1d):
                x = layer(x.transpose(1, 2)).transpose(1, 2).contiguous()

        output = self.fc(x)

        return {
            "log_probs": nn.functional.log_softmax(output, dim=-1),
            "log_probs_length": spectrogram_length,
        }
