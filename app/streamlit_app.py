import sys

sys.path.append("./")

import time

import streamlit as st
from buttons_action import add_diacritics, clear_input, clear_output, remove_diacritics


def main():
    """
    Main scene
    """
    new_title = '<p style="font-size: 30px;">Welcome to our Vietnamese Diacritics \
    Restoration App!</p>'
    title = st.markdown(new_title, unsafe_allow_html=True)

    read_me = st.markdown(
        """
    This project was built by using Streamlit and ...
                          """
    )

    st.sidebar.title("Select Activity")
    choices = st.sidebar.selectbox("MODE", ("About us", "App"))

    if choices == "App":
        title.empty()
        read_me.empty()
        add_diacritics_scene()


def add_diacritics_scene():
    """
    Usage: The main scene of the app.
    We can add accents by using models.
    It should be passed with model_name argument.
    """
    # Title
    title_text = '<p style="font-size: 36px;">Vietnamese Diacritics Restoration App</p>'
    st.markdown(title_text, unsafe_allow_html=True)

    # Add custom CSS styles
    st.markdown(
        """
        <style>
        /* Customize the text input box */
        .stTextInput, .stTextArea, .stSelectbox, .stNumberInput {
            # background-color: #f5f5f5;
            color: #333333;
            border: 1px solid #cccccc;
            border-radius: 10px;
            padding: 15px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Display the select box
    st.selectbox("Choose your model", ["Transformer", "N-Gram"], key="choice")
    if st.session_state.choice == "N-Gram":
        st.number_input(
            "Enter n-gram between 2 or 3", min_value=2, max_value=3, key="ngram"
        )
        st.number_input(
            "Enter beam search k from 1 to 15", min_value=1, max_value=15, key="k"
        )
    st.markdown("""-----""")

    with st.container():
        # Input text and output area
        st.text_input(
            label="Input",
            placeholder="Hom nay cong ty Cinnamon to chuc mot buoi party hoanh trang",
            key="input",
        )

        progress_bar = st.progress(0, text="progress bar")
        predict_status = st.success("This is a success message!", icon="✅")
        st.session_state.input_box_status = st.error(
            "This is an error message!", icon="🚨"
        )
        predict_status.empty()
        progress_bar.empty()
        st.session_state.input_box_status.empty()

        # Buttons
        col1, _, col2, _, col3 = st.columns([1, 1, 1, 1, 1])
        with col1:
            if st.button("Add diacritics"):
                # Check if input is empty
                if st.session_state.input.strip() == "":
                    st.session_state.input_box_status.error(
                        "Input cannot be empty", icon="🚨"
                    )
                else:
                    add_diacritics()
                    for percent_complete in range(100):
                        time.sleep(0.001)
                        progress_bar.progress(
                            percent_complete + 1,
                            # text=f"Time taken: {st.session_state.time}",
                        )
                        predict_status.success(
                            f"Completion time in {round(st.session_state.time, 3)}\
                                seconds.",
                            icon="✅",
                        )

        with col2:
            st.button("Clear input", on_click=clear_input)
        with col3:
            st.button("Strip diacritics", on_click=remove_diacritics)

        # Output
        st.text_area(
            "Output",
            key="output",
            placeholder="Hôm nay công ty Cinnamon tổ chức một buổi party hoành tráng",
        )

        # Clear output button
        _, _, col2, _, _ = st.columns([1, 1, 1, 1, 1])
        with col2:
            st.button(label="Clear output", on_click=clear_output)


if __name__ == "__main__":
    main()
