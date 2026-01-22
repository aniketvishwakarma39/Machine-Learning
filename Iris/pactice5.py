from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

iris=load_iris()
x=iris.data
y=iris.target

x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=200)
model.fit(x_train,y_train)

y_pred= model.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)

print(accuracy)
plt.scatter(x_test[:, 2], x_test[:, 3], c=y_pred)
plt.xlabel("petal length")
plt.ylabel("petal width")
plt.title("iris dataset accuracy")
plt.show()

