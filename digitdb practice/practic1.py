from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digit=load_digits()
x=digit.data
y=digit.target

plt.figure(figsize=(10,4))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(digit.images[i], cmap='gray')
    plt.title(f"label:{digit.target[i]}")
    plt.axis("off")
plt.title("567")
plt.show()