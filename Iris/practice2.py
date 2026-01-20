from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris= load_iris()
X=iris.data
Y= iris.target 

mask=Y<2
x_2=X[mask]
y_2=Y[mask]

plt.scatter(x_2[:,2],x_2[:,3], c=y_2)
plt.xlabel("petal length")
plt.ylabel("petal width")
plt.title("only 2 flowers")
plt.show()