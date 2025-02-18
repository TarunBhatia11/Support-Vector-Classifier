# Support Vector Classifier (SVC)

## Overview
This project implements a **Support Vector Classifier (SVC)** for classification tasks using Python. The model is trained on a dataset to classify different samples based on input features.

## Dataset
The dataset used in this notebook is `cell_samples.csv`, which contains labeled data for classification. The dataset undergoes preprocessing, exploratory data analysis (EDA), and feature selection before training the SVC model.

## Dependencies
To run this project, ensure you have the following Python libraries installed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Implementation Steps
1. **Import Libraries:** Load essential libraries such as `pandas`, `numpy`, `matplotlib`, and `seaborn` for data handling and visualization.
2. **Load Dataset:** Read the `cell_samples.csv` file and display its structure.
3. **Data Exploration:** Analyze dataset statistics, missing values, and class distributions.
4. **Data Preprocessing:** Handle missing values, encode categorical variables, and normalize data if necessary.
5. **Train-Test Split:** Split the dataset into training and testing sets.
6. **Model Training:** Train an SVC model using `sklearn.svm.SVC`.
7. **Model Evaluation:** Evaluate performance using metrics like accuracy, precision, recall, and confusion matrix.

## Usage
Run the Jupyter Notebook step by step to train and evaluate the SVC model. You can modify hyperparameters to optimize performance.

## Example Code Snippet
```python
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv('cell_samples.csv')
X = df.drop(columns=['target'])
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVC model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
```

## Results
After training, the model performance is evaluated using metrics like accuracy, precision, recall, and confusion matrix to assess classification effectiveness.

## Contribution
Feel free to modify the dataset, experiment with different SVM kernels (`linear`, `rbf`, `poly`), and contribute improvements to the project.

## Author
Tarun Bhatia


