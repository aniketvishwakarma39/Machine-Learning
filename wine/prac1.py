from sklearn.datasets import load_wine
import matplotlib.pyplot as plt

wine= load_wine()
x=wine.data
y=wine.target

print(wine.data)
print(wine.keys())
print(wine.feature_names)
print(wine.target_names)
print(wine.target)