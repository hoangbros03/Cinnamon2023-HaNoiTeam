import argparse
import pickle
import re
import time

from nltk.tokenize.treebank import TreebankWordDetokenizer

detokenize = TreebankWordDetokenizer().detokenize


def remove_vn_accent(word):
    """Remove diacritics function"""
    word = re.sub("[áàảãạăắằẳẵặâấầẩẫậ]", "a", word)
    word = re.sub("[éèẻẽẹêếềểễệ]", "e", word)
    word = re.sub("[óòỏõọôốồổỗộơớờởỡợ]", "o", word)
    word = re.sub("[íìỉĩị]", "i", word)
    word = re.sub("[úùủũụưứừửữự]", "u", word)
    word = re.sub("[ýỳỷỹỵ]", "y", word)
    word = re.sub("đ", "d", word)
    return word


# load model
with open(
    "/home/khanh/workspace/Cinnamon2023-HaNoiTeam/\
        ngram_model/checkpoints/2gram_model.pkl",
    "rb",
) as fin:
    model = pickle.load(fin)
    print("load model done")


def gen_accents_word(word):
    """Generate all possible accented words"""
    word_no_accent = remove_vn_accent(word.lower())
    all_accent_word = {word}
    for w in (
        open(
            "/home/khanh/workspace/Cinnamon2023-HaNoiTeam/ngram_model/vn_syllables.txt"
        )
        .read()
        .splitlines()
    ):
        w_no_accent = remove_vn_accent(w.lower())
        if w_no_accent == word_no_accent:
            all_accent_word.add(w)
    return all_accent_word


def beam_search(words, model, k=3):
    """Implement beam search algorithm"""
    sequences = []
    for idx, word in enumerate(words):
        if idx == 0:
            sequences = [([x], 0.0) for x in gen_accents_word(word)]
        else:
            all_sequences = []
            for seq in sequences:
                for next_word in gen_accents_word(word):
                    current_word = seq[0][-1]
                    score = model.logscore(next_word, [current_word])
                    new_seq = seq[0].copy()
                    new_seq.append(next_word)
                    all_sequences.append((new_seq, seq[1] + score))
            all_sequences = sorted(all_sequences, key=lambda x: x[1], reverse=True)
            sequences = all_sequences[:k]
    return sequences


def translate(sent, model_sent, k):
    """Predict and detokenize sentence"""
    sent = sent.replace("\n", "")
    result = beam_search(sent.lower().split(), model_sent, k)
    return detokenize(result[0][0])


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("sent", type=str)
    opt = parse.parse_args()

    start_time = time.time()
    result = translate(opt.sent, model, k=3)

    print("Input sentence: ", opt.sent)
    print("Predicted sentence: ", result)
