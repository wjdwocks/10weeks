import numpy as np
import matplotlib.pyplot as plt
from sklearn import (datasets, tree, ensemble, metrics)
from matplotlib.colors import ListedColormap
# decision tree는 if-else문처럼 각각의 구역을 나누어서 이쪽은 이거, 저쪽은 저거를 여러번 한다.

iris = datasets.load_iris()
iris.data = iris.data[:,0:2]
iris.feature_names = iris.feature_names[0:2]
iris.color = np.array([(1,0,0), (0,1,0), (0,0,1)])

model = tree.DecisionTreeClassifier(max_depth = 4)
model.fit(iris.data, iris.target)

x_min, x_max = iris.data[:, 0].min() - 1, iris.data[:,0].max()+1
y_min, y_max = iris.data[:,1].min() - 1, iris.data[:,1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
xy = np.vstack((xx.flatten(), yy.flatten())).T
zz = model.predict(xy)
plt.contourf(xx,yy,zz.reshape(xx.shape), cmap = ListedColormap(iris.color), alpha = 0.2)

predict = model.predict(iris.data)
accuracy = metrics.balanced_accuracy_score(iris.target, predict)

plt.title(f'Decision tree ({accuracy:.3f})')
plt.scatter(iris.data[:,0], iris.data[:,1], c=iris.color[iris.target], edgecolors=iris.color[predict])
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()

tree.plot_tree(model, feature_names=iris.feature_names, class_names=iris.target_names, impurity=False)
plt.show()