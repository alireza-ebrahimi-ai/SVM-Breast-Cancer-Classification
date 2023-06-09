import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a pipeline to normalize data and select the best features
preprocessor = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(f_classif, k=2))
])

# Fit the preprocessor to the training data and transform the test data
X_train_transformed = preprocessor.fit_transform(X_train, y_train)
X_test_transformed = preprocessor.transform(X_test)

# Get the indices of the selected features
selected_features = preprocessor.named_steps['feature_selection'].get_support(indices=True)

# Get the names of the selected features
feature_names = data.feature_names[selected_features]

# Create a mesh to plot the decision boundary
x_min, x_max = X_test_transformed[:, 0].min() - 1, X_test_transformed[:, 0].max() + 1
y_min, y_max = X_test_transformed[:, 1].min() - 1, X_test_transformed[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

fignum = 1

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Fit the model for different kernels
for i,kernel in enumerate(('linear', 'poly', 'rbf')):
    clf = SVC(kernel=kernel)
    clf.fit(X_train_transformed, y_train)

    # Make predictions on the transformed test data and calculate the accuracy
    y_pred = clf.predict(X_test_transformed)
    accuracy = accuracy_score(y_test, y_pred)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary and the data points using seaborn
    axs[i].set_title(f'Kernel: {kernel} - Accuracy: {accuracy*100:.2f}%')
    sns.set_style('whitegrid')
    sns.scatterplot(x=X_test_transformed[y_test == 0, 0], y=X_test_transformed[y_test == 0, 1], color='b', label='Benign', ax=axs[i])
    sns.scatterplot(x=X_test_transformed[y_test == 1, 0], y=X_test_transformed[y_test == 1, 1], color='r', label='Malignant', ax=axs[i])
    axs[i].contour(xx, yy, Z,
                colors=['k', 'k', 'k'],
                linestyles=['--', '-', '--'],
                levels=[-0.5, 0, 0.5])
    axs[i].legend()
    axs[i].set_xlabel(feature_names[0])
    axs[i].set_ylabel(feature_names[1])
    axs[i].set_xlim(xx.min(), xx.max())
    axs[i].set_ylim(yy.min(), yy.max())
    axs[i].set_xticks(())
    axs[i].set_yticks(())
    fignum += 1

plt.show()
