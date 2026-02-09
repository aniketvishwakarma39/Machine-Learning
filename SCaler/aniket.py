import pandas as pd 
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


data = pd.DataFrame({
    "Age": [18, 22, 25, np.nan, 35, 45, 50],
    "Salary": [15000, 25000, 30000, 40000, np.nan, 900000, 50000],
    "Purchased": [0, 0, 1, 1, 1, 1, 0]
})

x=data.drop("Purchased", axis=1)
y=data["Purchased"]
print(data.isnull().sum())

imputer=SimpleImputer(strategy='median')
impted=imputer.fit_transform(x)

imputed1=pd.DataFrame(impted, columns=x.columns)
print(imputed1)

scaler=RobustScaler()
sc=scaler.fit_transform(impted)

sc1=pd.DataFrame(sc,columns=x.columns)

new=pd.concat([sc1,y.reset_index(drop=True)],axis=1)
print(new)
