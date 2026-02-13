import pandas as pd

dat= pd.DataFrame({
    "hours":[1,2,3,4,5],
    "marks":[30,40,50,60,70]

})
print(dat.corr())
print(dat.cov())