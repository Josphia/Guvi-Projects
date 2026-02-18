import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="EDA: Query Page",
    page_icon="📈",
    layout="wide"
)

st.title("🔍 Property Query Page")

st.write("Here users can enter property details and get predictions.")