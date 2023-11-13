import numpy as np
import matplotlib.pyplot as plt
from sklearn import (datasets, svm, metrics)
from matplotlib.colors import ListedColormap

# SVM은 데이터들을 두 구역으로 나누어서()binary linear classifier해서 데이터를 구분함.
# 데이터가 lines이나 plane으로 구분되지 않을 때에는 kernel trick이라고 해서 차원을 하나 늘려서 구분하게 됨.

iris = datasets.load_iris()
iris.data = iris.data[:,0:2]
iris.feature_names = iris.feature_names[0:2]
iris.color = np.array([(1,0,0), (0,1,0), (0,0,1)])

models = [
    {'name' : 'svm.LinearSVC', 'obj' : svm.LinearSVC()},
    {'name' : 'svm.SVC(linear)', 'obj' : svm.SVC(kernel='linear')},
    {'name' : 'svm.SVC(poly,2)', 'obj' : svm.SVC(kernel='poly', degree=2)},
    {'name' : 'svm.SVC(poly,3)', 'obj' : svm.SVC(kernel='poly')},
    {'name' : 'svm.SVC(poly,4)', 'obj' : svm.SVC(kernel='poly', degree=4)},
    {'name' : 'svm.SVC(rbf)', 'obj' : svm.SVC(kernel='rbf')},
    {'name' : 'svm.SVC(rbf, $\gamma$=1)', 'obj' : svm.SVC(kernel='rbf', gamma=1)},
    {'name' : 'svm.SVC(rbf, $\gamma$=4)', 'obj' : svm.SVC(kernel='rbf', gamma=4)},
    {'name' : 'svm.SVC(rbf, $\gamma$=16)', 'obj' : svm.SVC(kernel='rbf', gamma=16)},
    {'name' : 'svm.SVC(rbf, $\gamma$=64)', 'obj' : svm.SVC(kernel='rbf', gamma=64)},
    {'name' : 'svm.SVC(sigmoid)', 'obj' : svm.SVC(kernel='sigmoid')},
    ]

x_min, x_max = iris.data[:, 0].min() - 1, iris.data[:,0].max()+1
y_min, y_max = iris.data[:,1].min() - 1, iris.data[:,1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
xy = np.vstack((xx.flatten(), yy.flatten())).T

for model in models:
    model['obj'].fit(iris.data, iris.target)
    
    predict = model['obj'].predict(iris.data)
    model['acc'] = metrics.balanced_accuracy_score(iris.target, predict)
    
    zz = model['obj'].predict(xy)
    plt.contourf(xx,yy,zz.reshape(xx.shape), cmap = ListedColormap(iris.color), alpha = 0.2)
    
    plt.title(model['name'] + f' ({model["acc"]:.3f})')
    plt.scatter(iris.data[:,0], iris.data[:,1], c=iris.color[iris.target], edgecolors=iris.color[predict])
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.show()
#asdsd