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

        self.rnn = None

        self.row_conv_layer = nn.Conv1d(in_channels=self.rnn_hidden_size * 2,
                                        out_channels=self.rnn_hidden_size * 2, kernel_size=3, padding=1)

        self.fc_2layers = nn.Sequential(
            nn.Linear(in_features=self.rnn_hidden_size * 2, out_features=512),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(512, n_tokens)
        )

    def forward(self, spectrogram, spectrogram_length, **batch):
        spectrogram = self.convs(spectrogram.unsqueeze(1))

        # batch_size, ch, h, w
        batch_size, ch, height, w = spectrogram.shape

        # batch_size, w, ch, h
        spectrogram = spectrogram.permute(0, 3, 1, 2)
        spectrogram = spectrogram.reshape(batch_size, w, ch * height)

        if self.rnn is None:
            self.rnn = nn.GRU(input_size=ch * height, hidden_size=self.rnn_hidden_size, num_layers=self.n_layers,
                              bidirectional=True, batch_first=True)

        spectrogram, h = self.rnn(spectrogram)

        # batch_size, f_dim, time
        spectrogram = spectrogram.permute(0, 2, 1)
        spectrogram = self.row_conv_layer(spectrogram)

        # batch_size, time, f_dim
        spectrogram = spectrogram.permute(0, 2, 1)

        spectrogram = self.fc_2layers(spectrogram)

        log_probs = nn.functional.log_softmax(spectrogram, dim=-1)

        return {
            "log_probs": log_probs,
            "log_probs_length": spectrogram_length // 2
        }
