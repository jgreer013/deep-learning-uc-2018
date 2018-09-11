from sklearn.datasets import fetch_lfw_people
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA #RandomizedPCA deprecated
from sklearn.model_selection import train_test_split #.cross_validation deprecated
from sklearn.model_selection import GridSearchCV #.grid_search deprecated
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set() # import seaborn for heatmap
import matplotlib.pyplot as plt


random_state = 42
faces = fetch_lfw_people(min_faces_per_person=60)

# Instantiate models
pca = PCA(n_components=150, whiten=True, random_state = random_state, svd_solver='randomized')
svc = SVC(kernel='rbf', class_weight='balanced')
model = make_pipeline(pca, svc)

x_train, x_test, y_train, y_test = train_test_split(faces.data, faces.target, random_state = random_state)

# Find optimal parameters
param_grid = {'svc__C': [1, 5, 10, 50],
              'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]}
grid = GridSearchCV(model, param_grid=param_grid)

grid.fit(x_train, y_train)

print(grid.best_params_)

# Pick best estimator and predict with it
model = grid.best_estimator_
yfit = model.predict(x_test)

# Display results
fig, ax = plt.subplots(4, 6)
for i, axis in enumerate(ax.flat):
    axis.imshow(x_test[i].reshape(62, 47), cmap='bone')
    axis.set(xticks=[], yticks=[])
    axis.set_ylabel(faces.target_names[yfit[i]].split()[-1],
                    color='black' if yfit[i] == y_test[i] else 'blue')
    
fig.suptitle('Predicted Names: Incorrect Labels in Blue', size=14)

plt.show()

# Show results of classification
print(classification_report(y_test, yfit,
                            target_names=faces.target_names))

# Create confusion matrix heat map for visualization
mat = confusion_matrix(y_test, yfit)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=faces.target_names,
            yticklabels=faces.target_names)
plt.xlabel('True Label')
plt.ylabel('Predicted Label')
plt.show()