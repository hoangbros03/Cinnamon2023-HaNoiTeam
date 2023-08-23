import os
import pickle
import re
from collections import defaultdict

import gdown
import streamlit as st
import torch
from nltk.tokenize.treebank import TreebankWordDetokenizer

from ngram_model.predict import beam_search
from utils.vocab_word import Vocab

from models.transformers.model import Transformer  # isort: skip


TGT_VOCAB_PATH = "utils/vocab/vn_words_tone.txt"
SRC_VOCAB_PATH = "utils/vocab/vn_words_notone.txt"
CHECKPOINTS_FOLDER = "./checkpoints"
TRANS_CHECKPOINT_PATH = "./checkpoints/model_best.pt"
TWO_GRAM_PATH = "./checkpoints/2gram_model.pkl"
THREE_GRAM_PATH = "./checkpoints/3gram_model.pkl"

TRANSFORMER_MODEL_ID = "1ea2Y06TV5zL_VMMtedoHIvvvAtDoT0bi"
TWO_GRAM_ID = "1zJ5YAvVcYlkZI7tcdcFIJeFqxZCyyVZ9"
THREE_GRAM_ID = "1oFQnwAyZFZRSYSKI3CVOelcMvzBIBZYS"


def download_checkpoint(file_id, destination):
    """Automatically download checkpoints from google drive"""
    if not os.path.exists(CHECKPOINTS_FOLDER):
        os.makedirs(CHECKPOINTS_FOLDER, exist_ok=True)
    gdown.download(
        f"https://drive.google.com/uc?id={file_id}", destination, quiet=False
    )


@st.cache_resource
def load_model_transformer():
    """Load model transformer"""
    # Load vocab
    tgt_vocab = Vocab(TGT_VOCAB_PATH)
    src_vocab = Vocab(SRC_VOCAB_PATH)

    # Model config
    d_model = 512
    n_head = 8
    n_hidden = 2048
    n_layer = 6
    device = "cuda:0"

    if not os.path.exists(TRANS_CHECKPOINT_PATH):
        print("Downloading transformer checkpoint file...")
        download_checkpoint(TRANSFORMER_MODEL_ID, TRANS_CHECKPOINT_PATH)
        print("Checkpoint file downloaded.")

    n_vocab_src = len(src_vocab)
    n_vocab_tgt = len(tgt_vocab)

    model = Transformer(
        n_vocab_src,
        n_vocab_tgt,
        d_model,
        n_hidden,
        n_head,
        n_layer,
        src_vocab.pad_id,
        device,
    )
    model.load_state_dict(
        torch.load(TRANS_CHECKPOINT_PATH, map_location=device)["model"]
    )
    model.to(device)
    model.eval()

    return model, tgt_vocab, src_vocab, device


@st.cache_resource
def load_model_2ngram():
    """Load model 2ngram"""
    # Load model
    detokenize = TreebankWordDetokenizer().detokenize
    if not os.path.exists(TWO_GRAM_PATH):
        print("Downloading 2-gram checkpoint file...")
        download_checkpoint(TWO_GRAM_ID, TWO_GRAM_PATH)
        print("Checkpoint file downloaded.")
    with open(
        TWO_GRAM_PATH,
        "rb",
    ) as fin:
        model = pickle.load(fin)
        print("load model done")

    return detokenize, model


@st.cache_resource
def load_model_3ngram():
    """Load model 3ngram"""
    # Load model
    detokenize = TreebankWordDetokenizer().detokenize
    if not os.path.exists(THREE_GRAM_PATH):
        print("Downloading 3-gram checkpoint file...")
        download_checkpoint(THREE_GRAM_ID, THREE_GRAM_PATH)
        print("Checkpoint file downloaded.")
    with open(
        THREE_GRAM_PATH,
        "rb",
    ) as fin:
        model = pickle.load(fin)
        print("load model done")

    return detokenize, model


def preprocess(utf8_str):
    """Preprocess input text by lowering, normalizing, storing punctuations"""
    punctuations = [
        ",",
        ";",
        ":",
        "'",
        "(",
        ")",
        "[",
        "]",
        '"',
        "-",
        "~",
        "/",
        "@",
        "{",
        "}",
        "*",
    ]
    # ending punctuation marks
    endPunctuations = [".", "?", "!", "..."]
    upperIndexes = defaultdict(list)
    puncIndexes = defaultdict(list)
    endPuncList = []
    spaceIndexes = defaultdict(list)
    sentences = re.split("(?<=[.!?]) +", utf8_str)
    processed_sentences = []
    num = len(sentences)
    for j in range(num):
        sent = sentences[j]
        for i in range(len(sent)):
            if sent[i].isupper():
                upperIndexes[j].append(i)
            elif sent[i] in punctuations:
                puncIndexes[j].append((i, sent[i]))
        for i in range(len(puncIndexes[j]) - 1, -1, -1):
            sent = sent[: puncIndexes[j][i][0]] + sent[puncIndexes[j][i][0] + 1 :]
        for i in range(len(sent)):
            if i > 0 and sent[i] == " " and sent[i - 1] == " ":
                spaceIndexes[j].append(i)
        for i in range(len(spaceIndexes[j]) - 1, -1, -1):
            sent = sent[: spaceIndexes[j][i]] + sent[spaceIndexes[j][i] + 1 :]
        sent = sent.lower()
        if sent[-1] in endPunctuations:
            endPuncList.append(sent[-1])
            sent = sent[:-1]
        else:
            endPuncList.append("")
        if sent[-1] == " ":
            sent = sent[:-1]
        processed_sentences.append(sent)
    return processed_sentences, upperIndexes, puncIndexes, endPuncList, spaceIndexes


def postprocess(sentences, upperIndexes, puncIndexes, endPuncList, spaceIndexes):
    """Postprocess output text: revert all punctuation marks, space and upper case"""
    num = len(sentences)
    output = ""
    for j in range(num):
        sent = sentences[j]
        while spaceIndexes[j]:
            ind = spaceIndexes[j].pop(0)
            sent = " ".join((sent[:ind], sent[ind:]))
        while puncIndexes[j]:
            ind, punc = puncIndexes[j].pop(0)
            sent = punc.join((sent[:ind], sent[ind:]))
        while upperIndexes[j]:
            ind = upperIndexes[j].pop(0)
            sent = sent[:ind] + sent[ind].upper() + sent[ind + 1 :]
        for i in range(len(sent) - 1, 0, -1):
            if sent[i] == " " and sent[i - 1] == " ":
                sent = sent[: i - 1] + sent[i:]
        if sent[-1] == " ":
            sent = sent[:-1]
        sent += endPuncList.pop(0)
        output += " " + sent
    return output


def predict_transfromer(sentences, model, tgt_vocab, src_vocab, device):
    """Predict pipeline by Transformer model
    Args:
        sentences: list of sentences
        model: Transformer model
        tgt_vocab: target vocab
        src_vocab: source vocab
        device: cuda or cpu
    Returns:
        predicted sentences
    """
    predicted_batch = []

    for sentence in sentences:
        src_tokens = src_vocab.encode(sentence.split()).unsqueeze(0).to(device)
        tgt_tokens_tensor = torch.tensor([tgt_vocab.sos_id]).unsqueeze(0).to(device)
        predicted_sentence = []
        ori_sentence = sentence.split()

        for i in range(src_tokens.shape[1]):
            with torch.no_grad():
                output = model(src_tokens, tgt_tokens_tensor)
                next_token = output.argmax(dim=-1)[:, -1]
            if next_token.item() == tgt_vocab.eos_id:  # End token
                break

            tgt_tokens_tensor = torch.cat(
                (tgt_tokens_tensor, next_token.unsqueeze(1)), dim=-1
            )
            predicted_token = tgt_vocab.decode(
                [next_token.squeeze(0).cpu().numpy().tolist()]
            )
            if predicted_token == "<unk>":
                ori_token = ori_sentence[i]
                predicted_sentence.append(ori_token)
            else:
                predicted_sentence.append(predicted_token)

        output = " ".join(predicted_sentence)
        predicted_batch.append(output)
    return predicted_batch


def predict_transformer_top_k(sentence, model, tgt_vocab, src_vocab, device, k=5):
    """Not done implemented yet!"""
    src_tokens = src_vocab.encode(sentence.split()).unsqueeze(0).to(device)
    tgt_tokens_tensor = torch.tensor([[tgt_vocab.sos_id]]).to(device)
    ori_sentence = sentence.split()

    for _ in range(src_tokens.shape[1]):
        with torch.no_grad():
            output = model(src_tokens, tgt_tokens_tensor)
            next_tokens_scores, next_tokens = torch.topk(output[:, -1], k)

        candidates = []
        for j in range(k):
            if next_tokens[0, j].item() == tgt_vocab.eos_id:
                break
            candidates.append((next_tokens[0, j], next_tokens_scores[0, j]))

        if not candidates:  # No valid candidates
            break

        candidate_tokens = [candidate[0].unsqueeze(0) for candidate in candidates]
        candidate_tokens = torch.cat(candidate_tokens, dim=0)
        tgt_tokens_tensor = torch.cat(
            [tgt_tokens_tensor.repeat(candidate_tokens.size(0), 1), candidate_tokens],
            dim=1,
        )

    predicted_sentences = []
    for candidate in tgt_tokens_tensor:
        predicted_sentence = []
        for token_id in candidate:
            token_id = token_id.item()
            if token_id == tgt_vocab.eos_id:
                break
            predicted_token = tgt_vocab.decode([[token_id]])[0]
            if predicted_token == "<unk>":
                ori_token = ori_sentence[len(predicted_sentence)]
                predicted_sentence.append(ori_token)
            else:
                predicted_sentence.append(predicted_token)
        output = " ".join(predicted_sentence)
        predicted_sentences.append(output)

    return predicted_sentences


def predict_ngram(sentences, detokenize, model, k=3):
    """Predict pipeline by ngram model
    Args:
        sentences: list of sentences
        detokenize: detokenize function
        model: ngram model
        k: beam search size
    Returns:
        predicted sentences
    """
    predicted_sentences = []
    for sentence in sentences:
        output = beam_search(sentence.split(), model, k)
        predicted_sentences.append(detokenize(output[0][0]))
    return predicted_sentences
