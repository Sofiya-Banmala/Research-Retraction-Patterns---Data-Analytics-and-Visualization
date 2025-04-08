import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from wordcloud import WordCloud

# Load the dataset
data = pd.read_csv('retractions35215.csv')

# Convert RetractionDate to datetime format with correct parsing
data['RetractionDate'] = pd.to_datetime(data['RetractionDate'], dayfirst=True, errors='coerce')

# Ensure RetractionDate is datetime-like
if pd.api.types.is_datetime64_any_dtype(data['RetractionDate']):
    print("RetractionDate successfully converted to datetime format.")
else:
    print("Error: RetractionDate is not in datetime format.")

# Exploratory Data Analysis (EDA)

## Top Journals by Retractions
top_journals = data['Journal'].value_counts().head(12)
plt.figure(figsize=(12, 6))
top_journals.plot(kind='bar', color='green')
plt.title('Top 10 Journals by Number of Retractions')
plt.xlabel('Journal')
plt.ylabel('Number of Retractions')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

## Distribution of Retractions by Subject
top_subjects = data['Subject'].value_counts().head(12)
plt.figure(figsize=(12, 8))
top_subjects.plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Retractions Across Top 10 Subjects')
plt.ylabel('')
plt.tight_layout()
plt.show()

## Top Countries by Retractions
country_data = data['Country'].str.split(';').explode()
top_countries = country_data.value_counts().head(12)
plt.figure(figsize=(12, 6))
top_countries.plot(kind='bar', color='green')
plt.title('Top 10 Countries by Number of Retractions')
plt.xlabel('Country')
plt.ylabel('Number of Retractions')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

## Correlation Heatmap of Numerical Data
numerical_data = data.select_dtypes(include=[np.number])
correlation_matrix = numerical_data.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
plt.title('Correlation Heatmap of Numerical Data')
plt.show()

# Clustering and PCA

## K-means Clustering and PCA
features = ['CitationCount', 'RetractionPubMedID', 'OriginalPaperPubMedID']
X = data[features].dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.5)
plt.title('Clusters on PCA-reduced Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster Label')
plt.show()

# Multiple Linear Regression
X = data.select_dtypes(include=['number']).dropna()
y = X.pop('CitationCount')

model = sm.OLS(y, sm.add_constant(X)).fit()

r_squared = model.rsquared
print(f"Coefficient of Determination (r-squared): {r_squared:.2f}")

p_values = model.pvalues
print("P-values for the coefficients:\n", p_values)

# Additional Visualizations

## Word Cloud of Retraction Reasons
reason_data = data['Reason'].str.split('; ').explode()
reason_counts = reason_data.value_counts().nlargest(10)

plt.figure(figsize=(16, 12))
reason_counts.plot(kind='barh', color='green')
plt.title('Top 10 Common Reasons for Retractions')
plt.xlabel('Frequency')
plt.ylabel('Reasons')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()

## Distribution of Retraction Years
plt.figure(figsize=(10, 6))
sns.histplot(data['RetractionDate'].dt.year.dropna())
plt.title('Distribution of Retraction Years')
plt.xlabel('Year')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.tight_layout(pad=3.0, w_pad=0.5, h_pad=1.0)
plt.show()


# Next, create a word cloud from the reasons field.
# Join all reasons into a single string, separating them with spaces
reasons_text = ' '.join(reason for reason in reason_data.dropna())

# Create the word cloud with specified dimensions and background color
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(reasons_text)

# Display the word cloud
plt.figure(figsize=(14, 7))  # Adjusted figure size for word cloud
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Remove the axis
plt.title('Word Cloud for Reasons of Retraction', fontsize=12)
plt.show()



# Convert the RetractionDate to datetime format with the correct date format
data['RetractionDate'] = pd.to_datetime(data['RetractionDate'], format='%d/%m/%Y')

# Extract year and month from RetractionDate for further analysis
data['Year'] = data['RetractionDate'].dt.year
data['Month'] = data['RetractionDate'].dt.month

# Yearly Trends: Number of Retractions Per Year
yearly_counts = data['Year'].value_counts().sort_index()
plt.figure(figsize=(10, 5))
plt.plot(yearly_counts.index, yearly_counts.values, marker='o', color='green',linestyle='-')
plt.title('Number of Retractions Per Year')
plt.xlabel('Year')
plt.ylabel('Number of Retractions')
plt.grid(True)
plt.show()

# Month/Season Analysis: Number of Retractions by Month
monthly_counts = data['Month'].value_counts().sort_index()
plt.figure(figsize=(10, 5))
plt.bar(monthly_counts.index, monthly_counts.values, color='green')
plt.title('Number of Retractions Per Month')
plt.xlabel('Month')
plt.ylabel('Number of Retractions')
plt.xticks(monthly_counts.index, ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(axis='y')
plt.show()



print("\nEnd of analysis Group 9.")
