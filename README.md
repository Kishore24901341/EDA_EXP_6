# EDA_EXP_6

## **Aim**

To perform complete Exploratory Data Analysis (EDA) on the Wine Quality dataset, detect and remove outliers using the IQR method, and compare the performance of a classification model (Logistic Regression) before and after outlier removal.

## **Algorithm**

1)Import pandas, numpy, seaborn, matplotlib, sklearn libraries.

## **NAME : KISHORE V**
## **REG NO : 212224240077**
## **Program**

```
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=';')

1 - DATA UNDERSTANDING
print("First 5 rows:\n", df.head())
print("\nDataset shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())

2 - UNIVARIATE ANALYSIS (HISTPLOTTING)
plt.figure(figsize=(15,4))
plt.subplot(1,3,1)
sns.histplot(df['alcohol'], kde=True)
plt.title("Alcohol Distribution")

plt.subplot(1,3,2)
sns.histplot(df['volatile acidity'], kde=True)
plt.title("Volatile Acidity Distribution")

plt.subplot(1,3,3)
sns.histplot(df['pH'], kde=True)
plt.title("pH Distribution")

plt.tight_layout()
plt.show()


3 - BIVARIATE ANALYSIS
import matplotlib.pyplot as plt
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.boxplot(x='quality', y='alcohol', data=df)
plt.title("Alcohol vs Quality")

plt.subplot(1,2,2)
sns.boxplot(x='quality', y='volatile acidity', data=df)
plt.title("Acidity vs Quality")

plt.tight_layout()
plt.show()

print("\nRelationship Explanation:")
print("- Higher quality wines tend to have higher alcohol levels.")
print("- Volatile acidity decreases as wine quality increases.")



4 - MULTIVARIATE ANALYSIS – CORRELATION

corr = df[['alcohol', 'volatile acidity', 'pH', 'quality']].corr()

plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

print("\nHighest Correlation with Quality:")
print(corr['quality'].sort_values(ascending=False))



5 - CLASSIFICATION – GOOD VS BAD WINE

df['good_wine'] = (df['quality'] >= 7).astype(int)

X = df.drop(['quality', 'good_wine'], axis=1)
y = df['good_wine']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


6 - OUTLIER DETECTION

features = ['alcohol', 'pH', 'volatile acidity']

plt.figure(figsize=(12, 4))

for i, feature in enumerate(features, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(y=df[feature])
    plt.title(f"{feature} Boxplot")

plt.tight_layout()
plt.show()
```

## **Output**

1 - DATA UNDERSTANDING

<img width="997" height="355" alt="image" src="https://github.com/user-attachments/assets/9d7f8c58-a763-4028-900e-6057e00c2ce3" />

<img width="497" height="117" alt="image" src="https://github.com/user-attachments/assets/01d54cd2-09f8-4ab6-abd4-3a5535704e92" />

2 - UNIVARIATE ANALYSIS (HISTPLOTTING)

<img width="1463" height="382" alt="image" src="https://github.com/user-attachments/assets/f8f4e194-1877-4d49-a2fd-be814df634c4" />

3 - BIVARIATE ANALYSIS

<img width="1239" height="380" alt="image" src="https://github.com/user-attachments/assets/553a71d7-2b85-4b66-8deb-0cb960c86d1c" />

4 - MULTIVARIATE ANALYSIS – CORRELATION

<img width="1092" height="479" alt="image" src="https://github.com/user-attachments/assets/fa1e17b0-ba08-4d86-9884-283c1eb7754f" />

5 - CLASSIFICATION – GOOD VS BAD WINE

<img width="599" height="116" alt="image" src="https://github.com/user-attachments/assets/a69bfef2-fd6d-43d8-a29c-597a00832587" />

6 - OUTLIER DETECTION

<img width="1162" height="375" alt="image" src="https://github.com/user-attachments/assets/9e56147d-0d01-4d9d-b8ac-e87f2ecb2b1f" />


## **Result**

Thus, To perform complete Exploratory Data Analysis (EDA) on the Wine Quality dataset, detect and remove outliers using the IQR method, and compare the performance of a classification model (Logistic Regression) before and after outlier removal has successfully completed.
