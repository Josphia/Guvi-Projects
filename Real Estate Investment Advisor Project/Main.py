import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(r"E:\VS Code Projects\Guvi-Projects\Real Estate Investment Advisor Project\india_housing_prices.csv")

#res= df.head()
#res = df.info()
#res = df.isnull().sum()

#df.drop_duplicates(inplace=True)
#res = df.info()

#print(df.describe())

#secondaryBackgroundColor="#ffc6c6"

le = LabelEncoder()

df['City_Encoded'] = le.fit_transform(df['City'])
df['Property_Type_Encoded'] = le.fit_transform(df['Property_Type'])

df['School_Density_Score'] = df['Nearby_Schools']
df.drop(columns=['Nearby_Schools'], inplace=True)
print(df.columns)