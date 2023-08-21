import re

import streamlit as st
import torch

from models.transformers.model import Transformer
from utils.vocab_word import Vocab


@st.cache_resource
def load_model():
    """Load model transformer"""
    # Load vocab
    tgt_vocab = Vocab("utils/vocab/tokenize_tone.txt")
    src_vocab = Vocab("utils/vocab/tokenize_notone.txt")

    # Model config
    d_model = 512
    n_head = 8
    n_hidden = 2048
    n_layer = 6
    device = "cuda:0"

    checkpoint_path = "checkpoints/model_best.pt"
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
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model"])
    model.to(device)
    model.eval()

    return model, tgt_vocab, src_vocab, device


def remove_diacritics():
    """
    Usage: Remove accents
    Return: Removed-accents input
    """
    input = st.session_state.input
    intab_l = "ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđ"
    intab_u = "ẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸĐ"
    intab = list(str(intab_l + intab_u))

    outtab_l = "a" * 17 + "o" * 17 + "e" * 11 + "u" * 11 + "i" * 5 + "y" * 5 + "d"
    outtab_u = "A" * 17 + "O" * 17 + "E" * 11 + "U" * 11 + "I" * 5 + "Y" * 5 + "D"
    outtab = outtab_l + outtab_u

    r = re.compile("|".join(intab))
    replaces_dict = dict(zip(intab, outtab))

    st.session_state.input = r.sub(lambda m: replaces_dict[m.group(0)], input)


def v_spacer(height, sb=False) -> None:
    """Add space between st components"""
    for _ in range(height):
        if sb:
            st.sidebar.write("\n")
        else:
            st.write("\n")


def clear_input():
    """Clear input box"""
    st.session_state.input = ""
    st.session_state.output = ""


def clear_output():
    """Clear output box"""
    st.session_state.output = ""


def add_diacritics():
    """Add diacritics using Transformer model"""
    model, tgt_vocab, src_vocab, device = load_model()
    sentence = st.session_state.input
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

    st.session_state.output = " ".join(predicted_sentence)