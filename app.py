"""Data Mining App: preprocessing + supervised and unsupervised learning in Streamlit.

Run with:  streamlit run app.py
"""
import streamlit as st

from src.data_loader import SUPPORTED_EXTENSIONS, load_dataset
from ui.preprocessing_page import render_preprocessing
from ui.supervised_page import RESULT_KEY as SUPERVISED_KEY
from ui.supervised_page import render_supervised
from ui.unsupervised_page import RESULTS_KEY as CLUSTER_KEY
from ui.unsupervised_page import render_unsupervised

st.set_page_config(page_title="Data Mining App", layout="wide")

load_cached = st.cache_data(show_spinner=False)(load_dataset)


def main() -> None:
    st.title(" Data Mining Automatation Tool")
    st.caption("Upload a dataset, preprocess it, then run supervised or unsupervised learning.")

    uploaded = st.file_uploader("Upload your dataset (.csv or .arff)", type=list(SUPPORTED_EXTENSIONS))
    if uploaded is None:
        st.info("Upload a file to get started. Sample datasets are available in `data/sample/`.")
        return

    content = uploaded.getvalue()
    try:
        df = load_cached(content, uploaded.name)
    except Exception as exc:
        st.error(f"Could not read {uploaded.name}: {exc}")
        return

    # A new dataset invalidates the results computed on the previous one.
    dataset_id = (uploaded.name, len(content))
    if st.session_state.get("dataset_id") != dataset_id:
        st.session_state.dataset_id = dataset_id
        st.session_state.pop(SUPERVISED_KEY, None)
        st.session_state.pop(CLUSTER_KEY, None)

    prep_tab, supervised_tab, unsupervised_tab = st.tabs(
        ["1 Preprocessing", "2 Supervised learning", "3 Unsupervised learning"]
    )
    with prep_tab:
        config = render_preprocessing(df)
    with supervised_tab:
        render_supervised(df, config)
    with unsupervised_tab:
        render_unsupervised(df, config)


main()
