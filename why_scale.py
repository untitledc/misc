import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def draw_svm(X, y, i):
    plt.subplot(1, 2, i)
    rbf_svc = svm.SVC(kernel='rbf', gamma=2**-7, C=2**13).fit(X, y)
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    #plt.subplots_adjust(wspace=0.4, hspace=0.4)
    Z = rbf_svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.imshow(Z, extent=(xx.min(), xx.max(), yy.min(), yy.max()), origin='lower',
               interpolation='nearest', cmap=plt.cm.Paired,alpha=0.5)
    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, s=40)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    #plt.title(titles[i])

    Z = rbf_svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.imshow(Z, extent=(xx.min(), xx.max(), yy.min(), yy.max()), origin='lower',
               interpolation='nearest', cmap=plt.cm.Paired,alpha=0.5)
    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, s=40)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

X_base = np.array([[0,0],[0,1],[1,0],[1,1],
                   [2,2],[2,3],[3,2],[3,3],
                   [1,3],[3,1],
                   [0,3],[3,0]], dtype='float_')
X1 = np.copy(X_base)
X2 = np.copy(X_base)
X1[:,1] = X_base[:,1]*2
X2[:,0] = X_base[:,0]*2
print(X1)
print(X2)
y = np.array([0,0,0,0,1,1,1,1,0,0,1,1])

h = .005  # step size in the mesh

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors

draw_svm(X1, y, 1)
draw_svm(X2, y, 2)

plt.savefig('test.png')


