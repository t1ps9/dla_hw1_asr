import gdown
import os


def download():
    gdown.download("https://drive.google.com/uc?id=1XdQqooHNgO9M35unPDLxjQwFf3MHEOEm")
    gdown.download("https://drive.google.com/uc?id=1OAvDuSfdgXOJ9FHsdo5CxTGZFiBL-FzN")
    gdown.download("https://drive.google.com/uc?id=1isFy_G-laxWCa9SYRTzDd_YtqFKfJXGr")

    os.mkdir('to_download')
    os.rename("model_best.pth", "to_download/model_best.pth")
    os.rename("librispeech-vocab.txt", "to_download/librispeech-vocab.txt")
    os.rename("3-gram.pruned.1e-7.arpa", "to_download/3-gram.pruned.1e-7.arpa")


if __name__ == "__main__":
    download()
