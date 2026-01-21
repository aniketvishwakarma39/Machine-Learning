from sklearn.datasets import load_breast_cancer 
import matplotlib.pyplot as plt
cancer = load_breast_cancer()
x=cancer.data 
y=cancer.target
print(cancer.feature_names)
print(cancer.target_names)
print(cancer.data)
print(cancer.target)