import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    # instance_data = {
    #         "audio": audio,
    #         "spectrogram": spectrogram,
    #         "text": text,
    #         "text_encoded": text_encoded,
    #         "audio_path": audio_path,
    #     }

    audio = [el['audio'].squeeze(0) if el['audio'].ndim > 1 else el['audio'] for el in dataset_items]
    audio = torch.nn.utils.rnn.pad_sequence(audio, batch_first=True)

    spectrogram_length = torch.tensor([el['spectrogram'].shape[2] for el in dataset_items])

    spectrogram = [el['spectrogram'].squeeze(0).permute(1, 0) for el in dataset_items]
    pad_spectrogram = torch.nn.utils.rnn.pad_sequence(spectrogram, batch_first=True).permute(0, 2, 1)

    text_encoded_length = torch.tensor([el['text_encoded'].shape[1] for el in dataset_items])
    text_encoded = [el['text_encoded'].squeeze(0) if el['text_encoded'].ndim
                    > 1 else el['text_encoded'] for el in dataset_items]
    pad_text_encoded = torch.nn.utils.rnn.pad_sequence(text_encoded, batch_first=True)

    text = [el['text'] for el in dataset_items]

    audio_path = [el['audio_path'] for el in dataset_items]

    result_batch = {
        "audio": audio,
        "spectrogram": pad_spectrogram,
        "spectrogram_length": spectrogram_length,
        "text": text,
        "text_encoded": pad_text_encoded,
        "text_encoded_length": text_encoded_length,
        "audio_path": audio_path,
    }

    return result_batch
