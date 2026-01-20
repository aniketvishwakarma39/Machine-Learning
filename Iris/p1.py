from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris=load_iris()
X=iris.data
Y=iris.target
plt.scatter(X[:,2], X[:,3], c=Y)
plt.xlabel("petal length")
plt.ylabel("Petal width")
plt.title("Iris data visualization")
plt.show()
print(iris.keys())
'''print(iris.data)
print(iris.target)'''
print(iris.data.shape)
print(iris.target.shape)
print(iris.feature_names)
print(iris.target_names)

