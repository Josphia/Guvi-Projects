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
df = pd.read_csv(r"E:\VS Code Projects\Guvi-Projects\Real Estate Investment Advisor Project\india_housing_prices.csv")

option = st.selectbox("", options = [
    "1. What is the distribution of property prices?", 
    "2. What is the distribution of property sizes?", 
    "3. How does the price per sq ft vary by property type?", 
    "4. Is there a relationship between property size and price?"], index=None, placeholder="Select One Josphia")


if option == "1. What is the distribution of property prices?":
    st.title("📊 EDA - Property Price Distribution")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot( df['Price_in_Lakhs'], bins=25, kde=True, ax=ax, color="#ffc6c6" )
    ax.set_title("Distribution of Property Prices")
    ax.set_xlabel("Price (in Lakhs)")
    ax.set_ylabel("Number of Properties")
    st.pyplot(fig)  

elif option == "2. What is the distribution of property sizes?":
    st.title("📊 EDA - Property Sizes Distribution")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot( df['Size_in_SqFt'], bins=50, kde=True, ax=ax, color="#ffc6c6" )
    ax.set_title("Distribution of Property Sizes")
    ax.set_xlabel("Size in SqFt")
    ax.set_ylabel("Number of Properties")
    st.pyplot(fig)  





