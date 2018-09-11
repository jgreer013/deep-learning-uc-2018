from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

log_reg = LogisticRegression()
soft_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10)
iris = datasets.load_iris()

print(iris["data"][0])
print(iris["target_names"])
print(iris["feature_names"])

X = iris["data"][:, 0:2]
y = (iris["target"][:] == 0).astype(np.int)
y2 = iris["target"]

log_reg.fit(X, y)
soft_reg.fit(X, y2)
X_new = np.linspace(0, 3, 1000).reshape(-1, 2)
y_probs = log_reg.predict_proba(X_new)
#plt.plot(X_new, y_probs[:, 1], "g-", label="Setosa")
#plt.plot(X_new, y_probs[:, 0], "b--", label="Not Setosa")

# Advanced plotting code based on source:
# https://github.com/ageron/handson-ml/blob/master/04_training_linear_models.ipynb

plt.figure(figsize=(10, 4))
plt.plot(X[y == 0, 0], X[y == 0, 1], "bs")
plt.plot(X[y == 1, 0], X[y == 1, 1], "g^")
plt.xlabel("Sepal length (cm)")
plt.ylabel("Sepal width (cm)")
left_right = np.array([4, 8.5])
boundary = -(log_reg.coef_[0][0] * left_right + log_reg.intercept_[0]) / log_reg.coef_[0][1]
plt.plot(left_right, boundary, "k--", linewidth=3)
plt.legend(["Not Setosa", "Setosa"])
plt.title("Logistic Regression of Setosa vs Non-Setosa")
plt.show()

plt.figure()

## Fancy Graph pre-reqs

# Creates a grid of the plot shown
x0, x1 = np.meshgrid(
        np.linspace(4, 8.5, 500).reshape(-1, 1),
        np.linspace(1.5, 5, 200).reshape(-1, 1),
    )

# Turns into a new X with the above grid
X_new = np.c_[x0.ravel(), x1.ravel()]

# Generates probabilties and predicted classes for X_new
y_proba = soft_reg.predict_proba(X_new)
y_predict = soft_reg.predict(X_new)

# Reshapes results
zz1 = y_proba[:, 1].reshape(x0.shape)
zz = y_predict.reshape(x0.shape)
##

plt.plot(X[y2 == 0, 0], X[y2 == 0, 1], "go", label="Setosa")
plt.plot(X[y2 == 1, 0], X[y2 == 1, 1], "bs", label="Versicolor")
plt.plot(X[y2 == 2, 0], X[y2 == 2, 1], "r^", label="Virginica")
plt.xlabel("Sepal length (cm)")
plt.ylabel("Sepal width (cm)")

# Fancy graph stuff from above link
from matplotlib.colors import ListedColormap
custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])

plt.contourf(x0, x1, zz, cmap=custom_cmap, linewidth=5)
contour = plt.contour(x0, x1, zz1, cmap=plt.cm.brg)
plt.clabel(contour, inline=1, fontsize=12)
plt.axis([4, 8.5, 1.5, 5])
plt.legend(loc="upper right", fontsize=14)
plt.title("Softmax Regression")
plt.show()
