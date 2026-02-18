import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(
    page_title="EDA: Query Page",
    page_icon="📈",
    layout="wide"
)

st.title("🔍 Property Query Page")

option = st.selectbox("", options = [
    "1. What is the distribution of property prices?", 
    "2. What is the distribution of property sizes?", 
    "3. How does the price per sq ft vary by property type?", 
    "4. Is there a relationship between property size and price?"], index=None, placeholder="Select One Josphia")

st.write(f"You have selected - {option}")