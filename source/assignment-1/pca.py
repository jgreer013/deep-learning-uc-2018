from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

pca = PCA(n_components=154)
mnist = fetch_mldata('MNIST original')
X, y = mnist["data"], mnist["target"]

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

sample_index = np.random.permutation(60000)
X_train, y_train = X_train[sample_index], y_train[sample_index]

X_red = pca.fit_transform(X_train)
X_rec = pca.inverse_transform(X_red)
rounded_X = np.around(X_red.copy(), 1) # round to 1 decimal
X_red_low_digits = pca.inverse_transform(rounded_X)



recons_index = np.random.randint(1, 60000) - 1

orig_digit = X_train[recons_index]
recons_digit = X_rec[recons_index]
low_recons_digit = X_red_low_digits[recons_index]

actual_num = y_train[recons_index]

orig_image = orig_digit.reshape(28, 28)
recons_image = recons_digit.reshape(28, 28)
low_recons_image = low_recons_digit.reshape(28,28)

print(actual_num)
print(X_red[recons_index])
print(rounded_X[recons_index])

fig = plt.figure()

orig_fig = fig.add_subplot(3,3,1)
plt.imshow(orig_image, cmap=matplotlib.cm.binary, interpolation="nearest")
orig_fig.set_title("Original")
plt.axis("off")

new_fig = fig.add_subplot(3,3,2)
plt.imshow(recons_image, cmap=matplotlib.cm.binary, interpolation="nearest")
new_fig.set_title("Reconstructed")
plt.axis("off")

low_fig = fig.add_subplot(3,3,3)
plt.imshow(low_recons_image, cmap=matplotlib.cm.binary, interpolation="nearest")
low_fig.set_title("Reconstructed (2 digits)")
plt.axis("off")

plt.show()

# Plot first 24 components
fig2, axes = plt.subplots(3, 8, figsize=(9, 4),
                                 subplot_kw={'xticks':[], 'yticks':[]},
                                 gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(pca.components_[i].reshape(28, 28), cmap='bone')

plt.show()
