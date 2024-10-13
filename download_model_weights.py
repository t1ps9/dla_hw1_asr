import gdown
import os


def download():
    gdown.download("https://drive.google.com/file/d/1XdQqooHNgO9M35unPDLxjQwFf3MHEOEm/view?usp=drive_link")
    gdown.download("https://drive.google.com/file/d/1OAvDuSfdgXOJ9FHsdo5CxTGZFiBL-FzN/view?usp=drive_link")
    gdown.download("https://drive.google.com/file/d/1isFy_G-laxWCa9SYRTzDd_YtqFKfJXGr/view?usp=drive_link")

    os.rename("model_best.pth", "data/other/model_best.pth")
    os.rename("librispeech-vocab.txt", "data/other/librispeech-vocab.txt")
    os.rename("3-gram.pruned.1e-7.arpa", "data/other/3-gram.pruned.1e-7.arpa")


if __name__ == "__main__":
    download()
