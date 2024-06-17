import streamlit as st
import io
import os
import yaml
import pyarrow
import tokenizers

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Setting page config to wide mode
st.set_page_config(layout="wide")

@st.cache_resource
def from_library():
    from retro_reader import RetroReader
    from retro_reader import constants as C
    return C, RetroReader

C, RetroReader = from_library()

my_hash_func = {
    io.TextIOWrapper: lambda _: None,
    pyarrow.lib.Buffer: lambda _: 0,
    tokenizers.Tokenizer: lambda _: None,
    tokenizers.AddedToken: lambda _: None
}

@st.cache_resource(hash_funcs=my_hash_func)
def load_en_electra_base_model():
    config_file = "configs/inference_en_electra_base.yaml"
    return RetroReader.load(config_file=config_file)

@st.cache_resource(hash_funcs=my_hash_func)
def load_en_electra_large_model():
    config_file = "configs/inference_en_electra_large.yaml"
    return RetroReader.load(config_file=config_file)


RETRO_READER_HOST = {
    "google/electra-base-discriminator": load_en_electra_base_model(),
    "google/electra-large-discriminator": load_en_electra_large_model(),
}

def main():
    # Sidebar Introduction
    st.sidebar.title("Welcome to Retro Reader")
    st.sidebar.write("""
    Explore the capabilities of state-of-the-art NLP models with Retro Reader. Select a model, type your query, and provide some context. See how the model interprets and processes your input to generate answers.
    """)

    st.title("Retrospective Reader Demo")
    st.markdown("## Model name")
    option = st.selectbox(
        label="Choose the model used in retro reader",
        options=(
            "[1] google/electra-base-discriminator",
            "[2] google/electra-large-discriminator"
        ),
        index=1,
    )
    lang_code, model_name = option.split(" ")
    retro_reader = RETRO_READER_HOST[model_name]

    lang_prefix = "EN"
    height = 200
    return_submodule_outputs = True
    
    st.markdown("## Demonstration")
    with st.form(key="my_form"):
        query = st.text_input(
            label="Type your query",
            value=getattr(C, f"{lang_prefix}_EXAMPLE_QUERY"),
            max_chars=None,
            help=getattr(C, f"{lang_prefix}_QUERY_HELP_TEXT"),
        )
        context = st.text_area(
            label="Type your context",
            value=getattr(C, f"{lang_prefix}_EXAMPLE_CONTEXTS"),
            height=height,
            max_chars=None,
            help=getattr(C, f"{lang_prefix}_CONTEXT_HELP_TEXT"),
        )
        submit_button = st.form_submit_button(label="Submit")
        
    if submit_button:
        with st.spinner("Please wait.."):
            outputs = retro_reader(query=query, context=context, return_submodule_outputs=return_submodule_outputs)
        answer, score = outputs[0]["id-01"], outputs[1]
        if not answer:
            answer = "No answer"
        st.markdown("## Results")
        st.write(answer)
        st.markdown("### Rear Verification Score")
        st.json(score)
        if return_submodule_outputs:
            score_ext, nbest_preds, score_diff = outputs[2:]
            st.markdown("### Sketch Reader Score (score_ext)")
            st.json(score_ext)
            st.markdown("### Intensive Reader Score (score_diff)")
            st.json(score_diff)
            st.markdown("### N Best Predictions (from intensive reader)")
            st.json(nbest_preds)

if __name__ == "__main__":
    main()
