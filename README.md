# ECG Heartbeat Categorization

## Overview
The `ECG Heartbeat Categorization.ipynb` notebook, available in this repository, focuses on identifying healthy heartbeats and anomalies in heartbeat signals using ECG (Electrocardiogram) data. The notebook employs a Random Forest Classifier for this purpose.

## Content

### Importing the Packages
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
```

### Loading the DataSet
```python
path = '/content/drive/MyDrive/data/ECG Heartbeat Categorization/'
normal_data = pd.read_csv(path + 'ptbdb_normal.csv', header=None)
abnormal_data = pd.read_csv(path + 'ptbdb_abnormal.csv', header=None)
train_data = pd.read_csv(path + 'mitbih_train.csv', header=None)
```

### Concatenating Datasets and Exploration
```python
train_data = pd.concat([normal_data, abnormal_data])
```

### Data Preprocessing
```python
# Split the data into features (X) and target labels (y)
X = train_data.iloc[:, :-1].values
y = train_data.iloc[:, -1].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### Building the Random Forest Classifier
```python
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)
```

### Model Evaluation
```python
# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```
**Accuracy:** 0.971830985915493

**Classification Report:**
```
              precision    recall  f1-score   support

         0.0       0.97      0.93      0.95      834
         1.0       0.97      0.99      0.98     2077

    accuracy                           0.97     2911
   macro avg       0.97      0.96      0.97     2911
weighted avg       0.97      0.97      0.97     2911
```

### Making Predictions
```python
y_pred = model.predict(X_test_scaled)
y_pred = (y_pred > 0.5).astype(int)
```

### Visualizing Feature Importance (if needed)
```python
feature_importance = classifier.feature_importances_
feature_names = [str(i) for i in range(X.shape[1])]
sns.barplot(x=feature_importance, y=feature_names)
plt.title("Feature Importance")
plt.show()
```

## Additional Notes
- The notebook achieved an accuracy of 97.18% in classifying healthy and anomalous heartbeats.
- Further predictions can be made using the trained model on new data.

---
