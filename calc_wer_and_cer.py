import argparse
from pathlib import Path

from src.metrics.utils import calc_cer, calc_wer
from src.text_encoder.ctc_text_encoder import CTCTextEncoder
from src.utils.io_utils import read_txt


import argparse
from pathlib import Path

from src.metrics.utils import calc_cer, calc_wer
from src.text_encoder.ctc_text_encoder import CTCTextEncoder


def main(predictions_dir, target_dir):
    wer_total, cer_total, count = 0, 0, 0

    for prediction_file in Path(predictions_dir).iterdir():
        if prediction_file.suffix == ".txt":
            prediction_id = prediction_file.stem

            target_file = Path(target_dir) / f"{prediction_id}.txt"

            if target_file.exists():
                predicted_text = read_txt(prediction_file)
                target_text = read_txt(target_file)

                target_text_norm = CTCTextEncoder.normalize_text(target_text)
                predicted_text_norm = CTCTextEncoder.normalize_text(predicted_text)

                cer = calc_cer(target_text_norm, predicted_text_norm)
                wer = calc_wer(target_text_norm, predicted_text_norm)

                cer_total += cer
                wer_total += wer
                count += 1

    if count > 0:
        print(f"Средний WER: {wer_total / count:.6f}")
        print(f"Средний CER: {cer_total / count:.6f}")
    else:
        print("Нет совпадений по id")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Вычисление WER и CER для предсказаний")
    parser.add_argument("--predictions_dir", required=True, type=str, help="Папка с предсказаниями")
    parser.add_argument("--target_dir", required=True, type=str, help="Папка с таргетами")
    args = parser.parse_args()

    main(args.predictions_dir, args.target_dir)
