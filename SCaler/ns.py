import pandas as pd 
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler , OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data= {
    "Experience": [1, 2, 3, 5, None, 10, 15, 20],
    "Income":     [25000, 28000, None, 45000, 30000, 80000, None, 500000],
    "Department": ["IT", "HR", None, "Finance", "HR", "IT", "Finance", "IT"],
    "Rating":     [3, 4, 5, None, 3, 5, 4, 5]
}

x= data.drop("Rating", axis=1)
y= data["Rating"]


x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.25, random_state=42)
n_features=x.select_dtypes(include=["float64", "int64"]).columns
c_features=x.select_dtypes(include=["object"]).columns

n_pipe=Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler())
])

c_pipe=Pipeline([
    ("encode", OneHotEncoder(sparse=False, handle_unknown="ignore") )
])

preprocess=ColumnTransformer([
    ('num', n_pipe, n_features),
    ("cat", c_pipe, c_features)
])

pipe=Pipeline([
    ("preprosess",preprocess),
    ("model", LogisticRegression())

])

pipe.fit(x_train, y_train)
y_pred=pipe.predict(x_test)

accuracy=accuracy_score(y_test, y_pred)

print(accuracy*100)