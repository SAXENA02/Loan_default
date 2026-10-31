"""
Credit Risk Assessment Portal (Streamlit, pure Python).
Run:  streamlit run streamlit_app.py
The portal loads the trained model directly, so it runs standalone.
"""
import streamlit as st

from ui import core, portal

st.set_page_config(page_title="Credit Risk Assessment Portal", page_icon="\U0001F3E6", layout="wide")

PAGES = {
    "Dashboard": portal.dashboard,
    "New Assessment": portal.assessment,
    "Portfolio Upload": portal.portfolio,
    "Data Stability": portal.stability,
    "Governance": portal.governance,
}

st.sidebar.markdown("### \U0001F3E6 Credit Risk Portal")
st.sidebar.text_input("Assessing officer", key="officer", placeholder="Name or staff ID (optional)")
choice = st.sidebar.radio("Navigate", list(PAGES), label_visibility="collapsed", key="nav")
st.sidebar.divider()
st.sidebar.caption("Decision-support prototype. Final credit decisions rest with an authorised officer.")
try:
    core.load_model()
    core.load_meta()
except Exception as e:  # e.g. a model file written by a different Python/OS
    st.error("The saved model could not be loaded on this computer.")
    st.write("Rebuild it once on this machine (takes about 2 to 3 minutes), then refresh this page:")
    st.code("python -m src.train_export")
    st.caption(str(e)[:200])
    st.stop()

PAGES[choice]()
