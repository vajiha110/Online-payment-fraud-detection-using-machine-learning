# Online-payment-fraud-detection-using-machine-learning
# Installing Required Libraries


```python
pip install pandas numpy matplotlib seaborn scikit-learn
```

    Requirement already satisfied: pandas in c:\users\sea farer\anaconda3\lib\site-packages (2.0.1)
    Requirement already satisfied: numpy in c:\users\sea farer\appdata\roaming\python\python310\site-packages (1.26.2)
    Requirement already satisfied: matplotlib in c:\users\sea farer\anaconda3\lib\site-packages (3.7.0)
    Requirement already satisfied: seaborn in c:\users\sea farer\anaconda3\lib\site-packages (0.12.2)
    Requirement already satisfied: scikit-learn in c:\users\sea farer\anaconda3\lib\site-packages (1.2.1)
    Requirement already satisfied: python-dateutil>=2.8.2 in c:\users\sea farer\anaconda3\lib\site-packages (from pandas) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in c:\users\sea farer\anaconda3\lib\site-packages (from pandas) (2022.7)
    Requirement already satisfied: tzdata>=2022.1 in c:\users\sea farer\anaconda3\lib\site-packages (from pandas) (2023.3)
    Requirement already satisfied: contourpy>=1.0.1 in c:\users\sea farer\anaconda3\lib\site-packages (from matplotlib) (1.0.5)
    Requirement already satisfied: cycler>=0.10 in c:\users\sea farer\anaconda3\lib\site-packages (from matplotlib) (0.11.0)
    Requirement already satisfied: fonttools>=4.22.0 in c:\users\sea farer\anaconda3\lib\site-packages (from matplotlib) (4.25.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in c:\users\sea farer\anaconda3\lib\site-packages (from matplotlib) (1.4.4)
    Requirement already satisfied: packaging>=20.0 in c:\users\sea farer\anaconda3\lib\site-packages (from matplotlib) (22.0)
    Requirement already satisfied: pillow>=6.2.0 in c:\users\sea farer\anaconda3\lib\site-packages (from matplotlib) (9.4.0)
    Requirement already satisfied: pyparsing>=2.3.1 in c:\users\sea farer\anaconda3\lib\site-packages (from matplotlib) (3.0.9)
    Requirement already satisfied: scipy>=1.3.2 in c:\users\sea farer\anaconda3\lib\site-packages (from scikit-learn) (1.10.0)
    Requirement already satisfied: joblib>=1.1.1 in c:\users\sea farer\anaconda3\lib\site-packages (from scikit-learn) (1.1.1)
    Requirement already satisfied: threadpoolctl>=2.0.0 in c:\users\sea farer\anaconda3\lib\site-packages (from scikit-learn) (2.2.0)
    Requirement already satisfied: six>=1.5 in c:\users\sea farer\anaconda3\lib\site-packages (from python-dateutil>=2.8.2->pandas) (1.16.0)
    Note: you may need to restart the kernel to use updated packages.
    

    WARNING: Ignoring invalid distribution -umpy (c:\users\sea farer\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -umpy (c:\users\sea farer\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -umpy (c:\users\sea farer\anaconda3\lib\site-packages)
    

#  Import Required Libraries and Load Dataset



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
data = pd.read_csv('C:/Users/Sea Farer/Downloads/online fraud detection dataset1.csv')

# Display the first few rows of the dataset
print(data.head())

```

       Unnamed: 0  step      type    amount     nameOrig oldbalanceOrg  \
    0           0     1   PAYMENT   9839.64  C1231006815        170136   
    1           1     1   PAYMENT   1864.28  C1666544295      21249. 0   
    2           2     1  TRANSFER    181.00  C1305486145           181   
    3           3     1  CASH OUT    181.00   C840083671           181   
    4           4     1   PAYMENT  11668.14  c2048537720      41554. 0   
    
       newbalanceOrig     nameDest  oldbalanceDest  newbalanceDest  isFraud  \
    0       160296.40  M1979787155               0               0        0   
    1        19384.72  M2044282225               0               0        0   
    2            0.00   C553264065               0               0        1   
    3            0.00    C38997010           21182               0        1   
    4        29885.86  M1230701703               0               0        0   
    
       isFIaggedFraud  
    0               0  
    1               0  
    2               0  
    3               0  
    4               0  
    

# Data Preprocessing and Cleaning


```python
# Check for missing values
print(data.isnull().sum())
```

    Unnamed: 0        0
    step              0
    type              0
    amount            0
    nameOrig          0
    oldbalanceOrg     0
    newbalanceOrig    0
    nameDest          0
    oldbalanceDest    0
    newbalanceDest    0
    isFraud           0
    isFIaggedFraud    0
    dtype: int64
    


```python
# Drop unnecessary columns (if any)
data = data.drop(['nameOrig', 'nameDest'], axis=1)
```


```python
# Convert 'type' column to numeric using One-Hot Encoding
data = pd.get_dummies(data, columns=['type'], drop_first=True)
```


```python
# Check data types and ensure all are numeric
print(data.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5 entries, 0 to 4
    Data columns (total 11 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   Unnamed: 0      5 non-null      int64  
     1   step            5 non-null      int64  
     2   amount          5 non-null      float64
     3   oldbalanceOrg   5 non-null      object 
     4   newbalanceOrig  5 non-null      float64
     5   oldbalanceDest  5 non-null      int64  
     6   newbalanceDest  5 non-null      int64  
     7   isFraud         5 non-null      int64  
     8   isFIaggedFraud  5 non-null      int64  
     9   type_PAYMENT    5 non-null      bool   
     10  type_TRANSFER   5 non-null      bool   
    dtypes: bool(2), float64(2), int64(6), object(1)
    memory usage: 498.0+ bytes
    None
    


```python
# Check data types and ensure all are numeric
print(data[['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']].dtypes)
```

    amount            float64
    oldbalanceOrg      object
    newbalanceOrig    float64
    oldbalanceDest      int64
    newbalanceDest      int64
    dtype: object
    


```python
# 1. Check for any non-numeric or improperly formatted values
for col in ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']:
    print(f"Unique values in {col}:")
    print(data[col].unique())
```

    Unique values in amount:
    [ 9839.64  1864.28   181.   11668.14]
    Unique values in oldbalanceOrg:
    ['170136' '21249. 0' '181' '41554. 0']
    Unique values in newbalanceOrig:
    [160296.4   19384.72      0.    29885.86]
    Unique values in oldbalanceDest:
    [    0 21182]
    Unique values in newbalanceDest:
    [0]
    


```python
# 2. Convert columns to numeric, forcing errors to NaN (if there are invalid entries)
for col in ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']:
    data[col] = pd.to_numeric(data[col], errors='coerce')
```


```python
# 3. Check for NaN values (which may result from conversion)
print("NaN values after conversion:")
print(data.isnull().sum())
```

    NaN values after conversion:
    Unnamed: 0        0
    step              0
    amount            0
    oldbalanceOrg     2
    newbalanceOrig    0
    oldbalanceDest    0
    newbalanceDest    0
    isFraud           0
    isFIaggedFraud    0
    type_PAYMENT      0
    type_TRANSFER     0
    dtype: int64
    


```python
# 4. Handle missing values (if any) by filling them with 0 (or use another strategy)
data = data.fillna(0)
```


```python
# 5. Confirm data types are now numeric
print(data[['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']].dtypes)
```

    amount            float64
    oldbalanceOrg     float64
    newbalanceOrig    float64
    oldbalanceDest      int64
    newbalanceDest      int64
    dtype: object
    


```python
# 6. Apply StandardScaler on the cleaned numeric columns
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data[['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']] = \
    scaler.fit_transform(data[['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']])

print("Scaling completed successfully.")
```

    Scaling completed successfully.
    

    C:\Users\Sea Farer\anaconda3\lib\site-packages\sklearn\utils\validation.py:767: FutureWarning: is_sparse is deprecated and will be removed in a future version. Check `isinstance(dtype, pd.SparseDtype)` instead.
      if not hasattr(array, "sparse") and array.dtypes.apply(is_sparse).any():
    C:\Users\Sea Farer\anaconda3\lib\site-packages\sklearn\utils\validation.py:605: FutureWarning: is_sparse is deprecated and will be removed in a future version. Check `isinstance(dtype, pd.SparseDtype)` instead.
      if is_sparse(pd_dtype):
    C:\Users\Sea Farer\anaconda3\lib\site-packages\sklearn\utils\validation.py:614: FutureWarning: is_sparse is deprecated and will be removed in a future version. Check `isinstance(dtype, pd.SparseDtype)` instead.
      if is_sparse(pd_dtype) or not is_extension_array_dtype(pd_dtype):
    C:\Users\Sea Farer\anaconda3\lib\site-packages\sklearn\utils\validation.py:767: FutureWarning: is_sparse is deprecated and will be removed in a future version. Check `isinstance(dtype, pd.SparseDtype)` instead.
      if not hasattr(array, "sparse") and array.dtypes.apply(is_sparse).any():
    C:\Users\Sea Farer\anaconda3\lib\site-packages\sklearn\utils\validation.py:605: FutureWarning: is_sparse is deprecated and will be removed in a future version. Check `isinstance(dtype, pd.SparseDtype)` instead.
      if is_sparse(pd_dtype):
    C:\Users\Sea Farer\anaconda3\lib\site-packages\sklearn\utils\validation.py:614: FutureWarning: is_sparse is deprecated and will be removed in a future version. Check `isinstance(dtype, pd.SparseDtype)` instead.
      if is_sparse(pd_dtype) or not is_extension_array_dtype(pd_dtype):
    

# Exploratory Data Analysis (EDA)


```python
# Check the distribution of the 'isFraud' column
sns.countplot(data['isFraud'])
plt.title('Distribution of Fraudulent and Non-Fraudulent Transactions')
plt.show()
```


    
![png](output_17_0.png)
    



```python
# Correlation matrix
plt.figure(figsize=(12,8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
```


    
![png](output_18_0.png)
    


# Train-Test Split


```python
# Define features and target variable
X = data.drop(['isFraud', 'isFIaggedFraud'], axis=1)
y = data['isFraud']
```


```python
# Split the data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```


```python
print(f'Training set size: {X_train.shape}')
print(f'Test set size: {X_test.shape}')
```

    Training set size: (4, 9)
    Test set size: (1, 9)
    

# Model Building: Random Forest Classifier


```python
# Initialize the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
```


```python
# Train the model
model.fit(X_train, y_train)
```

    C:\Users\Sea Farer\anaconda3\lib\site-packages\sklearn\utils\validation.py:767: FutureWarning: is_sparse is deprecated and will be removed in a future version. Check `isinstance(dtype, pd.SparseDtype)` instead.
      if not hasattr(array, "sparse") and array.dtypes.apply(is_sparse).any():
    C:\Users\Sea Farer\anaconda3\lib\site-packages\sklearn\utils\validation.py:605: FutureWarning: is_sparse is deprecated and will be removed in a future version. Check `isinstance(dtype, pd.SparseDtype)` instead.
      if is_sparse(pd_dtype):
    C:\Users\Sea Farer\anaconda3\lib\site-packages\sklearn\utils\validation.py:614: FutureWarning: is_sparse is deprecated and will be removed in a future version. Check `isinstance(dtype, pd.SparseDtype)` instead.
      if is_sparse(pd_dtype) or not is_extension_array_dtype(pd_dtype):
    C:\Users\Sea Farer\anaconda3\lib\site-packages\sklearn\utils\validation.py:605: FutureWarning: is_sparse is deprecated and will be removed in a future version. Check `isinstance(dtype, pd.SparseDtype)` instead.
      if is_sparse(pd_dtype):
    C:\Users\Sea Farer\anaconda3\lib\site-packages\sklearn\utils\validation.py:614: FutureWarning: is_sparse is deprecated and will be removed in a future version. Check `isinstance(dtype, pd.SparseDtype)` instead.
      if is_sparse(pd_dtype) or not is_extension_array_dtype(pd_dtype):
    




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestClassifier(random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestClassifier</label><div class="sk-toggleable__content"><pre>RandomForestClassifier(random_state=42)</pre></div></div></div></div></div>




```python
# Make predictions
y_pred = model.predict(X_test)
```

    C:\Users\Sea Farer\anaconda3\lib\site-packages\sklearn\utils\validation.py:767: FutureWarning: is_sparse is deprecated and will be removed in a future version. Check `isinstance(dtype, pd.SparseDtype)` instead.
      if not hasattr(array, "sparse") and array.dtypes.apply(is_sparse).any():
    C:\Users\Sea Farer\anaconda3\lib\site-packages\sklearn\utils\validation.py:605: FutureWarning: is_sparse is deprecated and will be removed in a future version. Check `isinstance(dtype, pd.SparseDtype)` instead.
      if is_sparse(pd_dtype):
    C:\Users\Sea Farer\anaconda3\lib\site-packages\sklearn\utils\validation.py:614: FutureWarning: is_sparse is deprecated and will be removed in a future version. Check `isinstance(dtype, pd.SparseDtype)` instead.
      if is_sparse(pd_dtype) or not is_extension_array_dtype(pd_dtype):
    

# Model Evaluation


```python
# Evaluate the model
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
```

    Confusion Matrix:
    [[1]]
    

    C:\Users\Sea Farer\anaconda3\lib\site-packages\sklearn\utils\validation.py:605: FutureWarning: is_sparse is deprecated and will be removed in a future version. Check `isinstance(dtype, pd.SparseDtype)` instead.
      if is_sparse(pd_dtype):
    C:\Users\Sea Farer\anaconda3\lib\site-packages\sklearn\utils\validation.py:614: FutureWarning: is_sparse is deprecated and will be removed in a future version. Check `isinstance(dtype, pd.SparseDtype)` instead.
      if is_sparse(pd_dtype) or not is_extension_array_dtype(pd_dtype):
    


```python
print('\nClassification Report:')
print(classification_report(y_test, y_pred))
```

    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00         1
    
        accuracy                           1.00         1
       macro avg       1.00      1.00      1.00         1
    weighted avg       1.00      1.00      1.00         1
    
    

    C:\Users\Sea Farer\anaconda3\lib\site-packages\sklearn\utils\validation.py:605: FutureWarning: is_sparse is deprecated and will be removed in a future version. Check `isinstance(dtype, pd.SparseDtype)` instead.
      if is_sparse(pd_dtype):
    C:\Users\Sea Farer\anaconda3\lib\site-packages\sklearn\utils\validation.py:614: FutureWarning: is_sparse is deprecated and will be removed in a future version. Check `isinstance(dtype, pd.SparseDtype)` instead.
      if is_sparse(pd_dtype) or not is_extension_array_dtype(pd_dtype):
    


```python
# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
```

    Accuracy: 100.00%
    

    C:\Users\Sea Farer\anaconda3\lib\site-packages\sklearn\utils\validation.py:605: FutureWarning: is_sparse is deprecated and will be removed in a future version. Check `isinstance(dtype, pd.SparseDtype)` instead.
      if is_sparse(pd_dtype):
    C:\Users\Sea Farer\anaconda3\lib\site-packages\sklearn\utils\validation.py:614: FutureWarning: is_sparse is deprecated and will be removed in a future version. Check `isinstance(dtype, pd.SparseDtype)` instead.
      if is_sparse(pd_dtype) or not is_extension_array_dtype(pd_dtype):
    

# Handling Imbalanced Data (Optional)


```python
from imblearn.over_sampling import SMOTE

# Apply SMOTE to the training data
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print(f'Resampled training set size: {X_train_res.shape}')
```

    Resampled training set size: (4, 9)
    

    C:\Users\Sea Farer\anaconda3\lib\site-packages\sklearn\utils\validation.py:605: FutureWarning: is_sparse is deprecated and will be removed in a future version. Check `isinstance(dtype, pd.SparseDtype)` instead.
      if is_sparse(pd_dtype):
    C:\Users\Sea Farer\anaconda3\lib\site-packages\sklearn\utils\validation.py:614: FutureWarning: is_sparse is deprecated and will be removed in a future version. Check `isinstance(dtype, pd.SparseDtype)` instead.
      if is_sparse(pd_dtype) or not is_extension_array_dtype(pd_dtype):
    C:\Users\Sea Farer\anaconda3\lib\site-packages\sklearn\utils\validation.py:767: FutureWarning: is_sparse is deprecated and will be removed in a future version. Check `isinstance(dtype, pd.SparseDtype)` instead.
      if not hasattr(array, "sparse") and array.dtypes.apply(is_sparse).any():
    C:\Users\Sea Farer\anaconda3\lib\site-packages\sklearn\utils\validation.py:605: FutureWarning: is_sparse is deprecated and will be removed in a future version. Check `isinstance(dtype, pd.SparseDtype)` instead.
      if is_sparse(pd_dtype):
    C:\Users\Sea Farer\anaconda3\lib\site-packages\sklearn\utils\validation.py:614: FutureWarning: is_sparse is deprecated and will be removed in a future version. Check `isinstance(dtype, pd.SparseDtype)` instead.
      if is_sparse(pd_dtype) or not is_extension_array_dtype(pd_dtype):
    

# Saving the Model (Optional)


```python
import joblib

# Save the model
joblib.dump(model, 'fraud_detection_model.pkl')

# Load the model (when needed)
loaded_model = joblib.load('fraud_detection_model.pkl')
```

# Making Predictions on New Data


```python
# Example of a new transaction (dummy data)
new_transaction = [[10000, 5000, 0, 5000, 0, 1, 0, 0, 10]] 

# Predict whether the transaction is fraudulent
fraud_prediction = model.predict(new_transaction)

if fraud_prediction == 1:
    print("The transaction is fraudulent.")
else:now
    print("The transaction is not fraudulent.")
```

    The transaction is fraudulent.
    

    C:\Users\Sea Farer\anaconda3\lib\site-packages\sklearn\base.py:420: UserWarning: X does not have valid feature names, but RandomForestClassifier was fitted with feature names
      warnings.warn(
    


```python
# Example of a new transaction (dummy data)
new_transaction = [[500, 2000, 1500, 3000, 3500, 0, 0, 5, 2]]

# Predict whether the transaction is fraudulent
fraud_prediction = model.predict(new_transaction)

if fraud_prediction[0] == 1:
    print("The transaction is fraudulent.")
else:
    print("The transaction is not fraudulent.")
```

    The transaction is not fraudulent.
    

    C:\Users\Sea Farer\anaconda3\lib\site-packages\sklearn\base.py:420: UserWarning: X does not have valid feature names, but RandomForestClassifier was fitted with feature names
      warnings.warn(
    


```python

```
