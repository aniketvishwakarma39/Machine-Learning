import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


data=pd.DataFrame({
    "salary":[25000,30000,35000,40000,10000000]
})

plt.figure()
plt.plot(data["salary"])
plt.title("outlier check")
plt.show()