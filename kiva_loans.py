import pandas as pd 
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg', 'Agg', etc.
import seaborn as sns


df = pd.read_csv("C:\\Users\\emana\\OneDrive\\Desktop\\BIProject\\masked_kiva_loans.csv")

## Handling nulls & duplicates 
print(df.isna().sum())
x = df['partner_id'].mean()
df['partner_id'].fillna(x, inplace=True)
print(df.isna().sum())
df.drop('borrower_genders', axis=1, inplace=True)
df.drop('country', axis=1, inplace=True)
print(df.duplicated().sum())

##outliers
plt.figure(figsize=(10, 5))
sns.boxplot(x=df['funded_amount'])
plt.title('Box Plot')
plt.show()
z_scores = np.abs(stats.zscore(df['funded_amount']))
outliers_z = df[(z_scores > 3)]
Q1 = df['funded_amount'].quantile(0.25)
Q3 = df['funded_amount'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = (df['funded_amount'] < lower_bound) | (df['funded_amount'] > upper_bound)
df.loc[df['funded_amount'] < lower_bound, 'funded_amount'] = Q1
df.loc[df['funded_amount'] > upper_bound, 'funded_amount'] = Q3
plt.figure(figsize=(10, 5))
sns.boxplot(x=df['funded_amount'])
plt.title('Box Plot')
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(x=df['loan_amount'])
plt.title('Box Plot')
plt.show()
z_scores = np.abs(stats.zscore(df['loan_amount']))
outliers_z = df[(z_scores > 3)]
Q1 = df['loan_amount'].quantile(0.25)
Q3 = df['loan_amount'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = (df['loan_amount'] < lower_bound) | (df['loan_amount'] > upper_bound)
df.loc[df['loan_amount'] < lower_bound, 'loan_amount'] = Q1
df.loc[df['loan_amount'] > upper_bound, 'loan_amount'] = Q3
plt.figure(figsize=(10, 5))
sns.boxplot(x=df['loan_amount'])
plt.title('Box Plot')
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(x=df['lender_count'])
plt.title('Box Plot')
plt.show()
z_scores = np.abs(stats.zscore(df['lender_count']))
outliers_z = df[(z_scores > 3)]
Q1 = df['lender_count'].quantile(0.25)
Q3 = df['lender_count'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = (df['lender_count'] < lower_bound) | (df['lender_count'] > upper_bound)
df.loc[df['lender_count'] < lower_bound, 'lender_count'] = Q1
df.loc[df['lender_count'] > upper_bound, 'lender_count'] = Q3
plt.figure(figsize=(10, 5))
sns.boxplot(x=df['lender_count'])
plt.title('Box Plot')
plt.show()



## Maping text values into numeric
print(df['sector'].value_counts())
df['sector']=df['sector'].map({'Agriculture':0,'Food':1,'Retail':2, 'Services':3, 'Personal Use':4, 'Housing':5,
                               'Clothing':6,'Education':7, 'Transportation':8, 'Arts':9, 'Health':10, 'Construction':11,
                               'Manufacturing':12, 'Entertainment':13, 'Wholesale':14})

print(df['repayment_interval'].value_counts())
df['repayment_interval']=df['repayment_interval'].map({'monthly':0,'irregular':1,'bullet':2, 'weekly':3})
print(df)

## statistics
print(df.describe())

##plots & visualization 
# histo
plt.hist(df['loan_amount'])
plt.xlabel('Loan Amount')
plt.ylabel('Frequency')
plt.title('Distribution of Loan Amounts')
plt.show()

# barc
plt.bar(df['sector'].value_counts().index, df['sector'].value_counts().values)
plt.xlabel('Sector')
plt.ylabel('Number of Loans')
plt.title('Distribution of Loans by Sector')
plt.show()

# scatter
plt.scatter(df['loan_amount'], df['repayment_interval'])
plt.xlabel('Loan Amount')
plt.ylabel('Repayment Interval (Encoded)')
plt.title('Loan Amount vs. Repayment Interval')
plt.show()

# pie
repayment_interval_counts = df['repayment_interval'].value_counts().sort_values(ascending=False)
plt.pie(repayment_interval_counts, labels=repayment_interval_counts.index, autopct="%1.1f%%")
plt.title('Distribution of Loans by Repayment Interval')
plt.show()

##Modelling
## Splitting The Data
y = df['funded_amount']
x = df.drop(columns=['funded_amount','date'], axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


## LinaerRegression Model
Model1 = LinearRegression()
Model1.fit(x_train,y_train)
y_pred = Model1.predict(x_test)
MSE = mean_squared_error(y_test,y_pred)
print(MSE)
R2 = r2_score(y_test,y_pred)
print(R2)

## Decision Tree Model
Model2 = DecisionTreeRegressor(max_depth=5)
Model2.fit(x_train, y_train)
y_pred = Model2.predict(x_test)
R2 = r2_score(y_test,y_pred)
print(R2)


