import pandas as pd

data=pd.DataFrame({
    "gender":["male","female","male", "female","male"],
    "purchase":[1,0,1,1,0]
})

print(pd.crosstab(data["gender"],data["purchase"]))
