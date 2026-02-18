import streamlit as st
import pandas as pd
import sqlite3

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏘️",
    layout="wide"
)

st.title("India Real Estate Investment Advisor")
st.write("Analyzing 250,000 property records...")