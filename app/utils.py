import streamlit as st
import unicodedata


def remove_diacritics():
    """
    Usage: Remove accents
    Return: Removed-accents input
    """
    input = st.session_state.input
    normalized_input = unicodedata.normalize("NFD", input)
    stripped_input = "".join(
        c for c in normalized_input if not unicodedata.combining(c)
    )
    st.session_state.input = stripped_input
    # return stripped_input


def v_spacer(height, sb=False) -> None:
    """Add space between st components"""
    for _ in range(height):
        if sb:
            st.sidebar.write("\n")
        else:
            st.write("\n")


def clear_input():
    """Clear input"""
    st.session_state.input = ""


def clear_output():
    """Clear output box text"""
    st.session_state.output = ""


# def add_diacritics(model):
# model = st.session_state.choice
# pass
