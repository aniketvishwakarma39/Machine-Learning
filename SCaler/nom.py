import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

data = pd.DataFrame({
    "Age": [18, 22, 25, np.nan, 35, 45, 50],
    "Salary": [15000, 25000, 30000, 40000, np.nan, 900000, 50000],
    "Purchased": [0, 0, 1, 1, 1, 1, 0]
})

x=data.drop("Purchased",axis=1)
y=data["Purchased"]

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.5, random_state=42)

f_pipe= Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("model", LogisticRegression())
])

f_pipe.fit(x_train, y_train)
y_pred= f_pipe.predict(x_test)
accuracy=accuracy_score(y_test, y_pred)
print(accuracy*100)

