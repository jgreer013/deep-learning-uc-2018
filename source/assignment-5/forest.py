from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_mldata
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

rand_forest = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)

mnist = fetch_mldata('MNIST original')
X, y = mnist["data"], mnist["target"]

rand_forest.fit(X, y)

fi = rand_forest.feature_importances_

fi_image = fi.reshape(28,28)

plt.imshow(fi_image, cmap=matplotlib.cm.hot, interpolation="nearest")
plt.axis("off")

color_bar = plt.colorbar(ticks=[fi.min(), fi.max()])
color_bar.ax.set_yticklabels(['Least Important', 'Most Important'])

plt.show()
