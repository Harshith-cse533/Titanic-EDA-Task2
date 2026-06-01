import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#Loading the Dataset
df = pd.read_csv("Titanic-Dataset.csv")
df.head()
#Check Dataset Information
df.info()
#to Show Statistics
df.describe()
#Checking Missing Values
df.isnull().sum()
#Create Histogram
df.hist(figsize=(10,8))
plt.show()
#to Create Boxplot
sns.boxplot(y=df["Fare"])
plt.show()
#to Create Heatmap
numeric_df = df.select_dtypes(include=['number'])

sns.heatmap(numeric_df.corr(), annot=True)
plt.show()
#to Create Pairplot
sns.pairplot(df[['Age','Fare','Pclass','Survived']])
plt.show()