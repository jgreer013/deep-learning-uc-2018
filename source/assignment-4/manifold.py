from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import *
from matplotlib import offsetbox
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Note: Plotting code based on code from following link:
# http://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html
# Code was modified to suite assignment (1/30th of data)
# Scale and visualize the embedding vectors
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(target[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 0.03:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imgs = data[i].reshape(28,28)
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(imgs, cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

mnist = fetch_mldata('MNIST original')

data = mnist.data[::30]

target = mnist.target[::30]

digits = mnist
y = target


plt.figure
iso = Isomap(n_components=2)
proj_iso = iso.fit_transform(data)
# Old Plotting code, kept for reference
#plt.scatter(proj_iso[:,0], proj_iso[:,1], c=target, cmap=plt.cm.get_cmap('jet', 10))
#plt.colorbar(ticks=range(10))
#plt.clim(-0.5, 9.5)
plot_embedding(proj_iso, 'Iso Embedding of digits')

plt.show()


plt.figure
lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
proj_lle = lle.fit_transform(data)
plot_embedding(proj_lle, 'LLE Embedding of digits')

plt.show()



plt.figure
mds = MDS(n_components=2)
proj_mds = mds.fit_transform(data)
plot_embedding(proj_mds, 'MDS Embedding of digits')

plt.show()



plt.figure
tsne = TSNE(n_components=2)
proj_tsne = tsne.fit_transform(data)
plot_embedding(proj_tsne, 'TSNE Embedding of digits')

plt.show()
