import streamlit as st
from utils import clear_input, remove_diacritics, clear_output


def main():
    new_title = '<p style="font-size: 30px;">Welcome to our Vietnamese Diacritics \
    Restoration App!</p>'
    title = st.markdown(new_title, unsafe_allow_html=True)

    read_me = st.markdown(
        """
    This project was built by using Streamlit and ...
                          """
    )

    st.sidebar.title("Select Activity")
    choices = st.sidebar.selectbox("MODE", ("About", "Main App"))

    if choices == "Main App":
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
        .stTextInput, .stTextArea, .stSelectbox {
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
    st.selectbox("Choose your model", ["Model_1", "Model_2", "Model_3"], key="choice")
    st.markdown("""-----""")
    # v_spacer(3)

    with st.container():
        # Input text and output area
        st.text_input(
            label="Please insert your word, sentence or paragraph that's need adding \
                diacritics",
            placeholder="Insert text",
            key="input",
        )

        # Buttons
        col1, _, col2, _, col3 = st.columns([1, 1, 1, 1, 1])
        with col1:
            st.button("Add diacritics")
        with col2:
            st.button("Clear input", on_click=clear_input)
        with col3:
            st.button("Strip diacritics", on_click=remove_diacritics)

        # Output
        st.text_area("Output", key="output")

        # Clear output button
        _, _, col2, _, _ = st.columns([1, 1, 1, 1, 1])
        with col2:
            st.button(label="Clear output", on_click=clear_output)


if __name__ == "__main__":
    main()
