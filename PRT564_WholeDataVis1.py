import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import IsolationForest
from sklearn.naive_bayes import CategoricalNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'retractions35215.csv'
data = pd.read_csv(file_path)

# Preprocessing
# Handle missing values
data.fillna('', inplace=True)

# Convert dates to numerical features (year) with the correct format
try:
    data['RetractionYear'] = pd.to_datetime(data['RetractionDate'], dayfirst=True, errors='coerce').dt.year
    data['OriginalPaperYear'] = pd.to_datetime(data['OriginalPaperDate'], dayfirst=True, errors='coerce').dt.year
except Exception as e:
    print(f"Error in date parsing: {e}")

# Verify date conversion
print(data[['RetractionDate', 'RetractionYear', 'OriginalPaperDate', 'OriginalPaperYear']].head())

# Reduce high cardinality for categorical columns
def reduce_cardinality(col, threshold=100):
    value_counts = col.value_counts()
    to_keep = value_counts.index[:threshold]
    return col.apply(lambda x: x if x in to_keep else 'Other')

data['Journal'] = reduce_cardinality(data['Journal'])
data['Publisher'] = reduce_cardinality(data['Publisher'])
data['Country'] = reduce_cardinality(data['Country'])
data['Author'] = reduce_cardinality(data['Author'])
data['Subject'] = reduce_cardinality(data['Subject'])
data['Institution'] = reduce_cardinality(data['Institution'])
data['Reason'] = reduce_cardinality(data['Reason'])

# Encode categorical variables
label_encoders = {}
categorical_columns = ['Journal', 'Publisher', 'Country', 'Author', 'RetractionNature', 'Reason', 'Paywalled', 'Subject', 'Institution', 'ArticleType']
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# Check the transformed data
print(data.head())

# Check the unique values of the target variable
print(f"Unique values in RetractionNature: {data['RetractionNature'].unique()}")

# If only one class in RetractionNature, choose a different target variable
if len(data['RetractionNature'].unique()) == 1:
    # Try using 'Reason' as the target variable instead
    target_variable = 'Reason'
else:
    target_variable = 'RetractionNature'

print(f"Using {target_variable} as the target variable for classification.")

# Select relevant features for PCA and clustering
features = ['Subject', 'Institution', 'Journal', 'Publisher', 'Country', 'Author', 'ArticleType', 
            'RetractionYear', 'OriginalPaperYear', 'RetractionNature', 'Reason', 'Paywalled', 'CitationCount']
X = data[features].values

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA of Retraction Data')
plt.show()

# K-means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.5)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('K-means Clustering of Retraction Data')
plt.show()

# Multiple Regression
# Predicting CitationCount
X_reg = data[['RetractionYear', 'OriginalPaperYear', 'Journal', 'Publisher', 'Country', 'RetractionNature', 'Reason', 'Paywalled']].values
y_reg = data['CitationCount'].values
X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Citation Count')
plt.ylabel('Predicted Citation Count')
plt.title('Actual vs Predicted Citation Count')
plt.show()

# Classification
# Predicting the chosen target variable
X_clf = data[['RetractionYear', 'OriginalPaperYear', 'Journal', 'Publisher', 'Country', 'Reason', 'Paywalled']].values
y_clf = data[target_variable].values

# Check the distribution of the target variable before split
print(f"Distribution of {target_variable} before split:")
print(data[target_variable].value_counts())

# Stratified split to maintain class distribution
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

# Check the distribution of the target variable after split
print(f"Distribution of {target_variable} in training set:")
print(pd.Series(y_train_clf).value_counts())

print(f"Distribution of {target_variable} in test set:")
print(pd.Series(y_test_clf).value_counts())

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier(random_state=42)
clf_rf.fit(X_train_clf, y_train_clf)
y_pred_clf_rf = clf_rf.predict(X_test_clf)
print("Random Forest Classifier:")
print(classification_report(y_test_clf, y_pred_clf_rf))

# Categorical Naive Bayes
clf_nb = CategoricalNB()
clf_nb.fit(X_train_clf, y_train_clf)
y_pred_clf_nb = clf_nb.predict(X_test_clf)
print("Categorical Naive Bayes:")
print(classification_report(y_test_clf, y_pred_clf_nb))

# Support Vector Machine (SVM)
clf_svm = SVC(random_state=42)
clf_svm.fit(X_train_clf, y_train_clf)
y_pred_clf_svm = clf_svm.predict(X_test_clf)
print("Support Vector Machine:")
print(classification_report(y_test_clf, y_pred_clf_svm))

# Outlier Detection
iso_forest = IsolationForest(contamination=0.05, random_state=42)
outliers = iso_forest.fit_predict(X_scaled)
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=outliers, cmap='coolwarm', alpha=0.5)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Outlier Detection in Retraction Data')
plt.show()
