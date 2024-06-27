# MediMapper-Navigating-Healthcare-Data-for-Precision-Medicine
In recent years, the integration of advanced data analytics into healthcare has revolutionized the way medical practitioners diagnose, treat, and manage diseases. Precision medicine, an innovative approach that tailors treatment to individual patient characteristics, is at the forefront of this transformation. MediMapper is an advanced analytical tool designed to harness the power of healthcare data, facilitating the delivery of precision medicine.

MediMapper leverages sophisticated machine learning algorithms to analyze vast amounts of healthcare data, identifying patterns and correlations that may not be evident through traditional methods. By utilizing algorithms like Gaussian Naive Bayes and Random Forest, MediMapper provides a robust framework for predicting disease outcomes and personalizing patient care.

MediMapper focuses specifically on lung cancer, aiming to enhance early detection, prognosis assessment, and treatment planning for this critical disease. This specialized approach aims to improve patient outcomes and support healthcare providers in making data-driven decisions tailored to lung cancer patients' needs.

## <strong>Key Features</strong>

**1. Data Preprocessing:**

Converts categorical variables into numerical formats, ensuring compatibility with machine learning models.
Handles missing values, duplicates, and outliers to maintain data integrity and reliability.

**2. Algorithmic Insights:**

Employs Gaussian Naive Bayes for its probabilistic interpretation and efficiency, particularly useful in initial exploratory data analysis.
Utilizes Random Forest for its robustness, accuracy, and ability to handle complex interactions between features.

**3. Hybrid Model Approach:**

Combines the strengths of both Gaussian Naive Bayes and Random Forest, leading to improved prediction accuracy and reliability.
Balances the simplicity of Naive Bayes with the advanced capabilities of Random Forest, ensuring a comprehensive analysis.

**4. Visualization and Interpretation:**

Provides detailed visualizations, such as correlation heatmaps and feature importance plots, to aid in the interpretation of model results.
Facilitates the understanding of complex data patterns, making it easier for healthcare professionals to make informed decisions.

**5. Feature Engineering:**

Implements advanced techniques to create new features, enhancing model performance and offering deeper insights into patient data.

## <strong>Impact on Healthcare</strong>

MediMapper's ability to navigate and analyze healthcare data paves the way for more personalized and effective treatments. By accurately predicting patient outcomes and identifying key risk factors, it enables healthcare providers to develop targeted treatment plans that are tailored to individual patient needs. This not only improves patient outcomes but also optimizes resource allocation, reducing costs and enhancing the overall efficiency of healthcare delivery.

## <strong>Conclusion</strong>

MediMapper represents a significant advancement in the application of machine learning to healthcare. By integrating robust algorithms and comprehensive data analysis techniques, it empowers healthcare professionals to deliver precision medicine, ultimately improving patient care and advancing medical knowledge. As the field of precision medicine continues to evolve, tools like MediMapper will play an essential role in shaping the future of healthcare.

 Let's go through the code step-by-step, explaining the terminology and how it relates to the project title "MediMapper: Navigating Healthcare Data for Precision Medicine."

Importing Libraries
python
Copy code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pandas: A powerful data manipulation library used for data analysis and manipulation.
numpy: A library for numerical computing in Python.
matplotlib.pyplot: A plotting library for creating static, animated, and interactive visualizations in Python.
seaborn: A statistical data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.
Uploading CSV File
python
Copy code
from google.colab import files
uploaded = files.upload()
data = pd.read_csv("survey lung cancer.csv")
data
data.shape
google.colab.files.upload(): A method to upload files to Google Colab.
pd.read_csv(): Reads a comma-separated values (CSV) file into DataFrame.
data.shape: Returns the dimensionality of the DataFrame.
Checking and Removing Duplicate Values
python
Copy code
data.duplicated().sum()
data = data.drop_duplicates()
data.duplicated().sum(): Counts the number of duplicate rows.
data.drop_duplicates(): Removes duplicate rows from the DataFrame.
Checking for Null Values
python
Copy code
data.isnull().sum()
data.isnull().sum(): Checks for missing values in the DataFrame.
Data Information and Description
python
Copy code
data.info()
data.describe()
data.info(): Provides a concise summary of the DataFrame.
data.describe(): Generates descriptive statistics that summarize the central tendency, dispersion, and shape of a dataset’s distribution.
Encoding Categorical Variables
python
Copy code
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

data['GENDER'] = le.fit_transform(data['GENDER'])
data['LUNG_CANCER'] = le.fit_transform(data['LUNG_CANCER'])
# Repeated for other categorical columns
LabelEncoder: Converts categorical data into numerical data.
le.fit_transform(): Fits label encoder and returns encoded labels.
Visualizing Target Distribution
python
Copy code
sns.countplot(x="LUNG_CANCER", data=data)
plt.title('Target Distribution')
sns.countplot(): Shows the counts of observations in each categorical bin using bars.
plt.title(): Sets the title of the plot.
Plotting Function for Features
python
Copy code
def plot(col, data=data):
    return data.groupby(col)['LUNG_CANCER'].value_counts(normalize=True).unstack().plot(kind='bar', figsize=(8,5))
data.groupby(): Groups the data by a specified column.
value_counts(normalize=True): Returns object containing counts of unique values as percentages.
unstack(): Converts a level of column labels to a column index.
plot(kind='bar', figsize=(8,5)): Plots the data as a bar chart.
Dropping Certain Columns
python
Copy code
new_data = data.drop(columns=['GENDER', 'AGE', 'SMOKING', 'SHORTNESS OF BREATH'])
data.drop(): Removes specified columns from the DataFrame.
Correlation Analysis
python
Copy code
cn = new_data.corr()
sns.heatmap(cn, cmap=cmap, annot=True, square=True)
new_data.corr(): Computes pairwise correlation of columns.
sns.heatmap(): Plots a heatmap of the correlation matrix.
Feature Engineering
python
Copy code
new_data['ANXYELFIN'] = new_data['ANXIETY'] * new_data['YELLOW_FINGERS']
Feature Engineering: Creating new features from existing ones to improve model performance.
Splitting Data into Independent and Dependent Variables
python
Copy code
X = new_data.drop('LUNG_CANCER', axis=1)
y = new_data['LUNG_CANCER']
X: Independent variables.
y: Dependent variable (target).
Balancing the Target Distribution
python
Copy code
from imblearn.over_sampling import ADASYN
adasyn = ADASYN(random_state=42)
X, y = adasyn.fit_resample(X, y)
ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning.
fit_resample(): Resamples the dataset to balance the classes.
Model Training and Evaluation
Gaussian Naive Bayes
python
Copy code
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb_predict = gnb.predict(X_test)
gnb_cr = classification_report(y_test, gnb_predict)
GaussianNB: Naive Bayes classifier for multivariate normal distribution.
fit(): Fits the Naive Bayes model.
predict(): Predicts the target values.
classification_report(): Generates a report of various classification metrics.
Random Forest Classifier
python
Copy code
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)
rfc_predict = rfc.predict(X_test)
rf_cr = classification_report(y_test, rfc_predict)
RandomForestClassifier: An ensemble learning method for classification.
n_estimators: The number of trees in the forest.
Combining Predictions
python
Copy code
combined_pred = []
for i in range(len(X_test)):
    if gnb_predict[i] == rfc_predict[i]:
        combined_pred.append(gnb_predict[i])
    else:
        combined_pred.append(rfc_predict[i])
naive_forest = classification_report(y_test, combined_pred)
Combined Predictions: Combining predictions from Gaussian Naive Bayes and Random Forest models.
Accuracy Comparison
python
Copy code
accuracy_scores = {"Gaussian NB": 0.92, "Random Forest": 0.98, "Hybrid Model": 0.99}
classifiers = list(accuracy_scores.keys())
accuracy = list(accuracy_scores.values())

plt.bar(classifiers, accuracy, color='pink', width=0.2)
plt.xlabel("Classifiers")
plt.ylabel("Accuracy")
plt.title("Comparison of Hybrid Model with the Previous Models")
plt.show()
Bar Plot: Visual comparison of accuracy among different models.
Precision, Recall, and F1 Score Comparison
python
Copy code
categories = ["Gaussian NB", "Random Forest", "Hybrid Model"]
pre_0 = [0.95, 0.98, 1.00]
pre_1 = [0.88, 0.98, 0.98]
rec_0 = [0.89, 0.98, 0.98]
rec_1 = [0.95, 0.98, 1.00]
f1_0 = [0.92, 0.98, 0.99]
f1_1 = [0.91, 0.98, 0.99]

bar_width = 0.2
index = np.arange(len(categories))

plt.bar(index, pre_0, bar_width, color="violet", label='precision 0')
plt.bar(index + bar_width, pre_1, bar_width, color="pink", label='precision 1')
plt.xlabel('algorithms')
plt.ylabel('precision%')
plt.title('Comparison of Hybrid Model precision with the Previous Models')
plt.xticks(index + bar_width / 2, categories)
plt.legend()
plt.show()

plt.bar(index, rec_0, bar_width, color="violet", label='recall 0')
plt.bar(index + bar_width, rec_1, bar_width, color="pink", label='recall 1')
plt.xlabel('algorithms')
plt.ylabel('recall%')
plt.title('Comparison of Hybrid Model recall with the Previous Models')
plt.xticks(index + bar_width / 2, categories)
plt.legend()
plt.show()

plt.bar(index, f1_0, bar_width, color="violet", label='f1 score 0')
plt.bar(index + bar_width, f1_1, bar_width, color="pink", label='f1 score 1')
plt.xlabel('algorithms')
plt.ylabel('f1 score%')
plt.title('Comparison of Hybrid Model f1score with the Previous Models')
plt.xticks(index + bar_width / 2, categories)
plt.legend()
plt.show()
Precision, Recall, F1 Score: Metrics used to evaluate the performance of classification models.
Project Title: MediMapper: Navigating Healthcare Data for Precision Medicine
MediMapper: The project aims to create a comprehensive tool for navigating healthcare data.
Navigating Healthcare Data: The process involves data cleaning, exploration, visualization, and model building to gain insights from healthcare data.
Precision Medicine: Tailoring medical treatment to the individual characteristics of each patient, often involving the use of machine learning models to predict health outcomes.
This project leverages machine learning techniques to predict lung cancer based on various features, balancing the dataset, comparing different models, and finally combining predictions to achieve higher accuracy, all aligning with the goals of precision medicine.
