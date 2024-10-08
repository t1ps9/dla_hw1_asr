import editdistance

# Based on seminar materials

# Don't forget to support cases when target_text == ''


def calc_cer(target_text, predicted_text) -> float:
    if len(target_text) == 0:
        return 1

    return editdistance.eval(target_text, predicted_text) / len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    if len(target_text) == 0:
        return 1
    print(target_text)
    print('------')
    print(predicted_text)
    return editdistance.eval(target_text.split(), predicted_text.split()) / len(target_text.split())
