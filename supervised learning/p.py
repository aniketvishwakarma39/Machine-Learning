import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

data={
    "hours":[1,2,3,4,5,6,7,8,9,10],
    "marks":[35,40,50,55,60,65,70,78,85,92]
}

df = pd.DataFrame(data)

x=df[["hours"]]
y=df["marks"]

x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=42)
model=LinearRegression()

model.fit(x_train, y_train)

y_pred= model.predict(x_test)

accuracy=r2_score(y_test,y_pred)
print("msq",mean_squared_error(y_test,y_pred))
print(accuracy)
print("predicted marks",y_pred)
print("Actual marks ",y_test.values)

p_hours=model.predict([[10]])
print(p_hours)

plt.scatter(x, y, label="Actual Data")
#plt.plot(x, model.predict(x), label="Regression Line")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Linear Regression (Hours vs Marks)")
plt.legend()
plt.show()