## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:

```python
import pandas as pd
df = pd.read_csv("EncodingData.csv")

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# Ordinal Encoding
pm = ['Hot', 'Warm', 'Cold']
el = OrdinalEncoder(categories=[pm])
el.fit_transform(df[["ord_2"]])
```

## Output: 
<img width="202" height="281" alt="image" src="https://github.com/user-attachments/assets/2f68a270-f2f3-4608-a673-3a0f20a11c7d" />

```python
df['bo2'] = el.fit_transform(df[['ord_2']])
df
```
## Output: 
<img width="561" height="410" alt="image" src="https://github.com/user-attachments/assets/4ec8709e-6f3e-4b25-94ee-679d231ac1c5" />

```python
# Label Encoding
le=LabelEncoder()
dfc = df.copy()
dfc['ord_2'] = le.fit_transform(dfc['ord_2'])
dfc
```
## Output: 
<img width="450" height="401" alt="image" src="https://github.com/user-attachments/assets/f2885de8-946b-4635-b044-ba1b84a8948f" />

```python
# One Hot Encoding
from sklearn.preprocessing import OneHotEncoder
oh = OneHotEncoder(sparse_output=False)
df2 = df.copy()
enc = pd.DataFrame(oh.fit_transform(df2[["nom_0"]]))
df2 = pd.concat([df2, enc], axis=1)
df2
```
## Output: 
<img width="563" height="414" alt="image" src="https://github.com/user-attachments/assets/fdb92d8a-5e92-4966-89e6-dac7c819b359" />


```python
pd.get_dummies(df2, columns=["nom_0"])
```
## Output: 
<img width="854" height="418" alt="image" src="https://github.com/user-attachments/assets/4980f272-79dc-4068-8f7e-cd16f88e1234" />

```python
#Binary Encoding
from category_encoders import BinaryEncoder
data = pd.read_csv("data.csv")

be = BinaryEncoder()
nd = be.fit_transform(data['Ord_2'])
dfb = pd.concat([data, nd], axis=1)
dfb
```
## Output: 
<img width="883" height="405" alt="image" src="https://github.com/user-attachments/assets/a9f4eb94-41d8-4bc0-9f35-148255bca585" />

```python
#Feature Transformation
from scipy import stats
import numpy as np
data = "Data_to_Transform.csv"
df = pd.read_csv(data)
df.skew()
```
## Output: 
<img width="397" height="159" alt="image" src="https://github.com/user-attachments/assets/c4fd60a2-fe32-406a-b20f-be9227c123ad" />

```python
# Log Transformation
np.log(df['Highly Positive Skew'])
```
## Output: 
<img width="643" height="332" alt="image" src="https://github.com/user-attachments/assets/527675e8-e899-49bd-97ca-cf6318a9c061" />

```python
# Reciprocal Transformation
np.reciprocal(df["Moderate Positive Skew"])
```
## Output: 
<img width="656" height="345" alt="image" src="https://github.com/user-attachments/assets/85fc617f-2741-497f-be09-2836edc7d425" />

```python
#Square root Transformation
np.sqrt(df["Highly Positive Skew"])

```
## Output: 
<img width="590" height="317" alt="image" src="https://github.com/user-attachments/assets/143e6d94-6fbe-4262-a6a9-bc491d17da0f" />

```python 
# Square Transformation
np.square(df["Highly Positive Skew"])
```
## Output: 
<img width="628" height="322" alt="image" src="https://github.com/user-attachments/assets/0ce2feed-a41e-4c61-9fcf-cffb3cc6d098" />

```python
# Power Transformations
# Box Cox
df["Highly Positive Skew_boxcox"], parameters = stats.boxcox(df["Highly Positive Skew"])
df
```
## Output: 
<img width="1244" height="476" alt="image" src="https://github.com/user-attachments/assets/8e44993b-9db5-4b0f-9c86-28f710c51e74" />

```python
# Yeo Johnson
df["Highly Negative Skew_yeojohnson"], parameters = stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
## Output: 
<img width="466" height="190" alt="image" src="https://github.com/user-attachments/assets/8f1c9deb-3c88-4a4b-b683-f2be2fa73a31" />

```python
df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
df
```
## Output: 
<img width="1496" height="484" alt="image" src="https://github.com/user-attachments/assets/a41c943c-6392-4c64-be3a-30a6db32aaa5" />

```python
# Quantile Transformation
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution='normal')
df["moderate_negative_skew_1"] = qt.fit_transform(df[["moderate_negative_skew"]])

df
```
## Output: 
<img width="1761" height="479" alt="image" src="https://github.com/user-attachments/assets/851f2359-2f94-4284-8270-270e6bd18da5" />

```python
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["moderate_negative_skew"], line ='45')
plt.show()
```
## Output: 
<img width="724" height="537" alt="image" src="https://github.com/user-attachments/assets/8cbfb025-8749-4e39-adfb-2c62ed8fd403" />

```python
sm.qqplot(np.reciprocal(df["moderate_negative_skew"]), line ='45')
plt.show()
```
## Output: 
<img width="715" height="539" alt="image" src="https://github.com/user-attachments/assets/59bd9913-7381-4a29-92db-3cbfabf0d227" />

```python
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution='normal', n_quantiles=891)
df["moderate_negative_skew"] = qt.fit_transform(df[["moderate_negative_skew"]])
sm.qqplot(df["moderate_negative_skew"], line='45')
plt.show()
```
## Output: 
<img width="702" height="541" alt="image" src="https://github.com/user-attachments/assets/68b91313-3c25-492f-b952-1a19821e6260" />

# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file  was performed successfully

       
