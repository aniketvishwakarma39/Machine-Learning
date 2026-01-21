from sklearn.datasets import load_wine
import matplotlib.pyplot as plt

wine=load_wine()
x=wine.data
y=wine.target

plt.scatter(x[:,0],x[:,9], c=y)
plt.xlabel("alcoholic")
plt.ylabel("color intensity")
plt.title("scatter plot wine datasets")
plt.show()