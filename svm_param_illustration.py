import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def get_fisher(data_matrix, y):
    global file_encoding

    X1 = data_matrix[y==1,:]
    X2 = data_matrix[y==0,:]
    ncol = X1.shape[1]
    fisher = []
    for i in range(ncol):
        x1 = X1[:,i]
        x2 = X2[:,i]
        f = (np.mean(x1) - np.mean(x2))**2 / (np.var(x1)**2 + np.var(x2)**2)
        fisher.append(f)

    reverse_index = list(reversed(np.argsort(fisher)))
    return (fisher, reverse_index)


# import some data to play with
dataset = datasets.load_breast_cancer()
X = dataset.data[:, [16,17]]  # we only take the first two features. We could
                      # avoid this ugly slicing by using a two-dim dataset
y = dataset.target
X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.9, random_state=42)
scaler = MinMaxScaler()
X = scaler.fit_transform(X_train)
y = y_train

h = .005  # step size in the mesh

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
rbf_svc = []
rbf_svc.append(svm.SVC(kernel='rbf', gamma=2**-7, C=2**13).fit(X, y))
rbf_svc.append(svm.SVC(kernel='rbf', gamma=2**3, C=2**13).fit(X, y))
rbf_svc.append(svm.SVC(kernel='rbf', gamma=2**-7, C=2).fit(X, y))
rbf_svc.append(svm.SVC(kernel='rbf', gamma=2**3, C=1).fit(X, y))

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# title for the plots
titles = ['C=2^13, gamma=2^-7',
          'C=2^13, gamma=2^3',
          'C=1, gamma=2^-7',
          'C=1, gamma=2^3']


for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    Z = rbf_svc[i].predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    print(Z.shape)
    plt.imshow(Z, extent=(xx.min(), xx.max(), yy.min(), yy.max()), origin='lower',
               interpolation='nearest', cmap=plt.cm.Paired,alpha=0.5)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, s=40)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(titles[i])

plt.savefig('test.png')

scaler2 = MinMaxScaler()
X = scaler2.fit_transform(dataset.data)
y = dataset.target
print(get_fisher(X,y))

