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
    "4. Is there a relationship between property size and price?",
    "5. Are there any outliers in price per sq ft or property size?",
    "6. What is the average price per sq ft by state?"
    ], index=None, placeholder="Select One Josphia")


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

elif option == "3. How does the price per sq ft vary by property type?":
    count = df['Property_Type'].value_counts()
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.pie(count, labels=count.index, autopct='%1.1f%%', startangle=90)
    ax.set_title("Distribution of Price Per Sqft Property Types")
    st.pyplot(fig)

elif option == "4. Is there a relationship between property size and price?":
    fig, ax = plt.subplots(figsize=(7,4))
    df_sample = df.sample(1000)
    sns.regplot(data=df_sample, x='Size_in_SqFt', y='Price_in_Lakhs', line_kws={'color':'green'}, color='#ffc6c6')
    ax.set_title("Property Size vs Price with Trend Line")
    ax.set_xlabel("Size (in Sqft)")
    ax.set_ylabel("Price (in Lakhs)")
    st.pyplot(fig)

elif option == "5. Are there any outliers in price per sq ft or property size?":
    st.subheader("Outlier Detection: Size in SqFt")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, y='Size_in_SqFt', ax=ax, color='lightgreen')
    ax.set_title("Outliers in Property Size")
    st.pyplot(fig)

elif option == "6. What is the average price per sq ft by state?":
    st.subheader("Average Price per Sqft by State")
    df6 = df.groupby('State')['Price_per_SqFt'].mean()
    st.dataframe(df6)



