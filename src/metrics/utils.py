import editdistance

# Based on seminar materials

# Don't forget to support cases when target_text == ''


def calc_cer(target_text, predicted_text) -> float:
    if len(target_text) == 0:
        return len(predicted_text)

    if len(predicted_text) == 0:
        return len(target_text)

    return editdistance.eval(target_text, predicted_text) / len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    if len(target_text) == 0:
        return len(predicted_text)

    if len(predicted_text) == 0:
        return len(target_text)

    if isinstance(target_text, str):
        target_text = target_text.split(' ')

    if isinstance(predicted_text, str):
        predicted_text = predicted_text.split(' ')

    return editdistance.eval(target_text, predicted_text) / len(target_text)
