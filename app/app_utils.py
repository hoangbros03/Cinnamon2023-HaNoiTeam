import os

import requests
import streamlit as st
from dotenv import load_dotenv

# Get .env info
load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL")


def remove_diacritics():
    """
    Usage: Remove accents
    Return: Removed-accents input
    """
    input = st.session_state.input
    response = requests.post(
        f"{BACKEND_URL}/remove", json={"string": input, "preprocessed": True}
    )
    print(response.json())
    st.session_state.input = response.json()["text_return"]
    # return stripped_input


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
    sentence = st.session_state.input
    response = requests.post(f"{BACKEND_URL}/predict", json={"string": sentence})
    st.session_state.output = response.json()["text_return"]
