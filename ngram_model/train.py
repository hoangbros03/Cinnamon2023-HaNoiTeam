import argparse
import pickle
import time

from nltk.lm import KneserNeyInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.tokenize import word_tokenize
from tqdm import tqdm


def get_arg():
    """Get training arguments"""
    parse = argparse.ArgumentParser()
    parse.add_argument("--data")
    parse.add_argument("--checkpoint")
    parse.add_argument("--ngram", type=int, default=3)
    return parse.parse_args()


def tokenize(doc):
    """Tokenize each sentence in dataset"""
    with open(
        "/home/khanh/workspace/Cinnamon2023-HaNoiTeam/ngram_model/vn_syllables.txt"
    ) as f:
        vnword = f.read().splitlines()
    result = []
    for sent in tqdm(doc):
        temp = word_tokenize(sent)
        for idx, word in enumerate(temp):
            if word not in vnword:
                temp[idx] = "unknown"
        result.append(temp)
    print("tokenize done")
    return result


if __name__ == "__main__":
    arg = get_arg()

    # Get train data and tokenize
    with open(arg.data, "r", encoding="utf-8") as fin:
        doc = fin.readlines()
    corpus = tokenize(doc)
    del doc

    vi_model = KneserNeyInterpolated(arg.ngram)
    train_data, padded_sent = padded_everygram_pipeline(arg.ngram, corpus)
    del corpus
    start_time = time.time()
    vi_model.fit(train_data, padded_sent)
    print("Train %s-gram model in %d s" % (arg.ngram, time.time() - start_time))
    print("Length of vocab = %s" % (len(vi_model.vocab)))

    with open(arg.checkpoint, "wb") as fout:
        pickle.dump(vi_model, fout)
    print("Save model successfully!")
