from pathlib import Path
import torchaudio

from src.datasets.base_dataset import BaseDataset


class CustomDirAudioDataset(BaseDataset):
    def __init__(self, audio_dir, transcription_dir=None, *args, **kwargs):
        data = []
        for path in Path(audio_dir).iterdir():
            entry = {}
            if path.suffix in [".mp3", ".wav", ".flac", ".m4a"]:
                entry["path"] = str(path)
                if transcription_dir and Path(transcription_dir).exists():
                    transc_path = Path(transcription_dir) / (path.stem + ".txt")
                    if transc_path.exists():
                        with transc_path.open() as f:
                            entry["text"] = f.read().strip()
                else:
                    entry["text"] = ""

                entry["audio_len"] = torchaudio.info(
                    entry["path"]).num_frames / torchaudio.info(entry["path"]).sample_rate
            if len(entry) > 0:
                data.append(entry)
        super().__init__(data, *args, **kwargs)
