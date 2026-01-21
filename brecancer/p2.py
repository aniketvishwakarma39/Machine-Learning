from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt 

cancer= load_breast_cancer()
x=cancer.data
y=cancer.target

plt.scatter(x[:,0],x[:,1], c=y)
plt.xlabel("radius")
plt.ylabel("texture")
plt.title("breast cancer")
plt.show()