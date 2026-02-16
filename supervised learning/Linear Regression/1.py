import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sk


data = pd.DataFrame({
    "Experience": [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
    "Salary":     [12000, 15000, 17000, 19000, 21000,23000, 25500, 28000, 30000, 33000]
})

print(data)

x=data["Experience"]
y=data["salary"]

x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.25, random_state=42)

model=LinearRegression()

model.fit(x_train, y_train)
