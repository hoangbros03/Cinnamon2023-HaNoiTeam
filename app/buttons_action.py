import re
import time
import streamlit as st
from app_utils import (
    load_model_2ngram,
    load_model_3ngram,
    load_model_transformer,
    predict_ngram,
    predict_transfromer,
    preprocess,
    postprocess,
)


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
    """Add function using Transformer and ngram model"""
    detokenize_2, ngram_2_model = load_model_2ngram()
    detokenize_3, ngram_3_model = load_model_3ngram()
    transfomer_model, tgt_vocab, src_vocab, device = load_model_transformer()

    sentence, upperIndexes, puncIndexes, endPuncList, spaceIndexes = preprocess(
        st.session_state.input
    )
    # output = predict_transformer_top_k(
    #     sentence, transfomer_model, tgt_vocab, src_vocab, device
    # )
    # for i, sent in enumerate(output):
    #     print(f"Top {i+1} prediction:", sent)

    if st.session_state.choice == "N-Gram":
        start_time = time.time()
        if st.session_state.ngram == 2:
            output = predict_ngram(
                sentence, detokenize_2, ngram_2_model, k=int(st.session_state.k)
            )
        if st.session_state.ngram == 3:
            output = predict_ngram(
                sentence, detokenize_3, ngram_3_model, k=int(st.session_state.k)
            )
        st.session_state.time = time.time() - start_time
    else:
        start_time = time.time()
        output = predict_transfromer(
            sentence, transfomer_model, tgt_vocab, src_vocab, device
        )
        st.session_state.time = time.time() - start_time

    print("Preprocessed input: ", sentence)
    print("Output sent: ", output)
    st.session_state.output = postprocess(
        output, upperIndexes, puncIndexes, endPuncList, spaceIndexes
    )
