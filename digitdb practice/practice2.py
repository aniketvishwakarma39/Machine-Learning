from sklearn.datasets import load_digits
digits=load_digits()
x=digits.data
y=digits.target

print(digits.data.shape)
print(digits.target.shape)
print(digits.keys())
print(digits.feature_names)
print(digits.target_names)
print(digits.data)
print(digits.target)