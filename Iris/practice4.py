from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris=load_iris()

x=iris.data
y=iris.target

mask=y!=1

x2=x[mask]
y2=y[mask]

plt.scatter(x2[:,2],x2[:,3],c=y2)
plt.xlabel("petal length")
plt.ylabel("petal width")
plt.title("setosa and virginica")
plt.show()