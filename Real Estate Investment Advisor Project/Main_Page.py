import streamlit as st
import pandas as pd
import sqlite3

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="📈",
    layout="wide"
)

st.title("India Real Estate Investment Advisor")
st.write("Analyzing 250,000 property records...")

df = pd.read_csv(r"E:\VS Code Projects\Guvi-Projects\Real Estate Investment Advisor Project\india_housing_prices.csv")

#st.dataframe(df)
