import numpy as np
from matplotlib import pyplot as plt
import mglearn


# generate deta set
X, y = mglearn.datasets.make_forge()

# plot data set
mglearn.discrete_scatter (X[:,0], X[:,1], y)
plt.legend(["Class0",  "Class1"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
print ("X.shape: {}".format(X.shape))


X, y = mglearn.datasets.make_wave (n_samples=40)
plt.plot (X, y, 'o')
plt.ylim (-3,3)
plt.xlabel ("Feature")
plt.ylabel ("Target")


from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print ("cancer.keys(): \n{}".format(cancer.keys()))


print ("Shape of cancer data: {}".format (cancer.data.shape))


print ("Sample counts per class: \n{}".format (
    {n: v for n, v in zip (cancer.target_names, np.bincount(cancer.target))}))


print ("Feature names: \n{}".format(cancer.feature_names))


from sklearn.datasets import load_boston

boston = load_boston()
print ("Data shape: {}".format(boston.data.shape))


X, y = mglearn.datasets.load_extended_boston ()
print ("X.shape: {}".format(X.shape))


mglearn.plots.plot_knn_classification(n_neighbors=1)


mglearn.plots.plot_knn_classification(n_neighbors=3)


from sklearn.model_selection import train_test_split

X, y = mglearn.datasets.make_forge()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)


from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier (n_neighbors=3)


clf.fit(X_train, y_train)


print ("Test set predictions: {}".format(clf.predict(X_test)))


print ("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))


print (clf.get_params())


fig, axes = plt.subplots(1, 3, figsize=(10, 3))

# zip: [a] -> [b] -> [(a,b)]
for n_neighbors, ax in zip([1, 3, 9], axes):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X,y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{} neighbor(s)".format(n_neighbors))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
    
axes[0].legend(loc=3)


# 汎化性能のプロット
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=66)

training_accuracy = []
test_accuracy = []
# n_neighbors を1から10まで試す
neighbors_settings = range(1,11)

for n_neighbors in neighbors_settings:
    # モデルを構築
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    training_accuracy.append(clf.score(X_train, y_train))
    test_accuracy.append(clf.score(X_test, y_test))
    
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()


# k-近傍回帰

from sklearn.metrics import euclidean_distances
from sklearn.neighbors import KNeighborsRegressor
from mglearn.plot_helpers import cm3

def my_plot_knn_regression(n_neighbors=1, ax=None):
    if ax==None:
        ax = plt.gca()
        
    X, y = mglearn.datasets.make_wave(n_samples=40)
    X_test = np.array([[-1.5], [0.9], [1.5]])

    dist = euclidean_distances(X, X_test)
    closest = np.argsort(dist, axis=0)

    reg = KNeighborsRegressor(n_neighbors=n_neighbors).fit(X, y)
    y_pred = reg.predict(X_test)

    for x, y_, neighbors in zip(X_test, y_pred, closest.T):
        for neighbor in neighbors[:n_neighbors]:
                ax.arrow(x[0], y_, X[neighbor, 0] - x[0], y[neighbor] - y_,
                          head_width=0, fc='k', ec='k')

    train, = ax.plot(X, y, 'o', c=cm3(0))
    test, = ax.plot(X_test, -3 * np.ones(len(X_test)), '*', c=cm3(2),
                     markersize=20)
    pred, = ax.plot(X_test, y_pred, '*', c=cm3(0), markersize=20)
    ax.vlines(X_test, -3.1, 3.1, linestyle="--")
    ax.legend([train, test, pred],
               ["training data/target", "test data", "test prediction"],
               ncol=3, loc=(.1, 1.025))
    ax.set_ylim(-3.1, 3.1)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")

fig, axes = plt.subplots(1,3,figsize=(15,4))

for n_neighbors, ax in zip ([1,3,9], axes):
    my_plot_knn_regression(n_neighbors=n_neighbors, ax=ax)


X, y = mglearn.datasets.make_wave(n_samples=40)

# wave データセットを訓練セットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# 3つの最近傍点を考慮するように設定してモデルのインスタンスを生成
reg = KNeighborsRegressor(n_neighbors=3)
# 訓練データと訓練ターゲットを用いてモデルを学習させる
reg.fit(X_train, y_train)


print("Test set prodictions:\n{}".format(reg.predict(X_test)))


print('Test set R^2: {:.2f}'.format(reg.score(X_test, y_test)))


fig, axes = plt.subplots(1, 3, figsize=(15,4))
# -3 から3 までの間に1,000天のデータポイントを作る
line = np.linspace(-3, 3, 1_000).reshape(-1,1)
for n_neighbors, ax in zip ([1, 3, 9], axes):
    # 1, 3, 9 近傍点で予測
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train, y_train)
    ax.plot(line, reg.predict(line))
    ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
    ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)
    
    ax.set_title("{} neighbor(s)\n train score: {:.2f} test score: {:.2f}".format(
            n_neighbors, reg.score(X_train, y_train), reg.score(X_test, y_test))) 
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
    
axes[0].legend(["Model predictions", "Training data/target", "Test data/target"], loc="best")


mglearn.plots.plot_linear_regression_wave()


from sklearn.linear_model import LinearRegression
X, y = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

lr = LinearRegression().fit(X_train, y_train)


print("lr.coef_: {}".format(lr.coef_))              # coefficient (w)
print("lr.intercept_: {}".format(lr.intercept_))    # intercept   (b)


# 訓練セットとテストセットに対する性能
# 単純すぎるモデルなので適合不足している
print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))


X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train, y_train)


# 過学習している
print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))


from sklearn.linear_model import Ridge

ridge = Ridge().fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))


ridge10 = Ridge(alpha=10).fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge10.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge10.score(X_test, y_test)))


ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge01.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge01.score(X_test, y_test)))


# 正則化パラメータを変えることによる検証
alphas = 10**np.arange(-3,2.25,0.25)
train_scores = np.zeros_like(alphas)
test_scores = np.zeros_like(alphas)

for idx, alpha in enumerate(alphas):
    ridge_local = Ridge(alpha=alpha).fit(X_train, y_train)
    train_scores[idx] = ridge_local.score(X_train, y_train)
    test_scores[idx] = ridge_local.score(X_test, y_test)
    
plt.plot(alphas, train_scores)
plt.plot(alphas, test_scores)
plt.xscale('log')


plt.plot(ridge.coef_, 's', label="Ridge alpha=1")
plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
plt.plot(ridge01.coef_, '^', label="Ridge alpha=0.1")

plt.plot(lr.coef_, 'o', label="LinearRegression")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.hlines(0, 0, len(lr.coef_))
plt.ylim(-25, 25)
plt.legend()


# training set size を変えてスコアがどう変化するか
# sklearn.model_selection.learning_curve を用いてplotしている
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html
mglearn.plots.plot_ridge_n_samples()


from IPython.display import Image, display_png
display_png(Image("20160706172820.png"))     # https://jojoshin.hatenablog.com/entry/2016/07/06/180923


from sklearn.linear_model import Lasso

lasso = Lasso().fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso.coef_ get_ipython().getoutput("= 0)))")


lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso001.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso001.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso001.coef_ get_ipython().getoutput("= 0)))")


lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso00001.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso00001.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso00001.coef_ get_ipython().getoutput("= 0)))")


# 正則化パラメータを変えることによる検証
alphas = 10**np.arange(-3,2.25,0.25)
train_scores = np.zeros_like(alphas)
test_scores = np.zeros_like(alphas)
n_features = np.zeros_like(alphas)

for idx, alpha in enumerate(alphas):
    lasso_local = Lasso(alpha=alpha, max_iter=100000).fit(X_train, y_train)
    train_scores[idx] = lasso_local.score(X_train, y_train)
    test_scores[idx] = lasso_local.score(X_test, y_test)
    n_features[idx] = np.sum(lasso_local.coef_ get_ipython().getoutput("= 0)")
    
fig, axes = plt.subplots(1,2,figsize=(10,4))
axes[0].plot(alphas, train_scores)
axes[0].plot(alphas, test_scores)
axes[0].set_xscale('log')
axes[0].set_ylabel("score")

axes[1].plot(alphas, n_features)
axes[1].set_xscale('log')
axes[1].set_ylabel("# of features")


plt.plot(lasso.coef_, 's', label="Lasso alpha=1")
plt.plot(lasso001.coef_, '^', label="Lasso alpha=0.01")
plt.plot(lasso00001.coef_, 'v', label="Lasso alpha=0.00001")

plt.plot(ridge01.coef_, 'o', label="Ridge alpha=0.1")
plt.legend(ncol=2, loc=(0,1.05))
plt.ylim(-25, 25)
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")


# Elastic net
# 正則化パラメータを変えることによる検証
from sklearn.linear_model import ElasticNet

alphas = 10**np.arange(-5,2.25,0.25)
train_scores = np.zeros_like(alphas)
test_scores = np.zeros_like(alphas)
n_features = np.zeros_like(alphas)

for idx, alpha in enumerate(alphas):
    elastic_local = ElasticNet(alpha=alpha, max_iter=100000).fit(X_train, y_train)
    train_scores[idx] = elastic_local.score(X_train, y_train)
    test_scores[idx] = elastic_local.score(X_test, y_test)
    n_features[idx] = np.sum(elastic_local.coef_ get_ipython().getoutput("= 0)")
    
fig, axes = plt.subplots(1,2,figsize=(10,4))
axes[0].plot(alphas, train_scores)
axes[0].plot(alphas, test_scores)
axes[0].set_xscale('log')
axes[0].set_ylabel("score")

axes[1].plot(alphas, n_features)
axes[1].set_xscale('log')
axes[1].set_ylabel("# of features")


# Logistic regression and support vector machine (support vector classifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

X, y = mglearn.datasets.make_forge()

fig, axes = plt.subplots(1, 2, figsize=(10, 3))

for model, ax in zip ([LinearSVC(), LogisticRegression()], axes):
    clf = model.fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5, ax=ax, alpha=.7)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{}".format(clf.__class__.__name__))
    ax.set_xlabel("Feature 0")
    ax.set_xlabel("Feature 1")
    
axes[0].legend()


# 正則化パラメータCを変えた場合
mglearn.plots.plot_linear_svc_regularization()


# cancer data set
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)
logreg = LogisticRegression().fit(X_train, y_train)
print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))


logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
print("Training set score {:.3f}".format(logreg100.score(X_train, y_train)))
print("Test set score {:.3f}".format(logreg100.score(X_test, y_test)))


logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
print("Training set score {:.3f}".format(logreg001.score(X_train, y_train)))
print("Test set score {:.3f}".format(logreg001.score(X_test, y_test)))


plt.plot(logreg.coef_.T, 'o', label='C=1')
plt.plot(logreg100.coef_.T, '^', label='C=100')
plt.plot(logreg001.coef_.T, 'v', label='C=0.01')
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.ylim(-5, 5)
plt.xlabel("Feature")
plt.ylabel("Coefficient magnitude")
plt.legend()


# 正則化パラメータを変えることによる検証
# warining supression
# https://qiita.com/idontwannawork/items/86c5b833cdc0a4cf58b5
Cs = 10**np.arange(-3,5.05,0.05)
train_scores = np.zeros_like(Cs)
test_scores = np.zeros_like(Cs)

for idx, C in enumerate(Cs):
    logreg_local = LogisticRegression(C=C, solver='liblinear').fit(X_train, y_train)
    train_scores[idx] = logreg_local.score(X_train, y_train)
    test_scores[idx] = logreg_local.score(X_test, y_test)
    
plt.plot(Cs, train_scores, label='train score')
plt.plot(Cs, test_scores, label='test score')
plt.xscale('log')
plt.legend(loc=4)


# L1-regularized Logistic Regression
# max_iter was changed
for C, marker in zip ([0.001, 1, 100], ['o', '^', 'v']):
    lr_l1 = LogisticRegression(C=C, penalty="l1", solver='liblinear', max_iter=1000).fit(X_train, y_train)
    print("Training accuracy of l1 logreg with C={:.3f}: {:.2f}".format(C, lr_l1.score(X_train, y_train)))
    print("Test accuracy of l1 logreg with C={:.3f}: {:.2f}".format(C, lr_l1.score(X_test, y_test)))
    plt.plot(lr_l1.coef_.T, marker, label="C={:.3f}".format(C))
    
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.xlabel("Feature")
plt.ylabel("Coefficient magnitude")

plt.ylim (-5, 5)
plt.legend(loc=5)


# データセット
from sklearn.datasets import make_blobs

X, y = make_blobs(random_state=42)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(["Class 0", "Class 1", "Class 2"])


linear_svm = LinearSVC().fit(X, y)
print("Coefficient shape:", linear_svm.coef_.shape)
print("Intercept shape:", linear_svm.intercept_.shape)


mglearn.discrete_scatter(X[:,0],X[:,1],y)
line = np.linspace(-15,15)
for coef, intercept, color in zip (linear_svm.coef_, linear_svm.intercept_, ['b', 'r', 'g']):
    plt.plot(line, -(line * coef[0] + intercept)/ coef[1], c=color)
    
plt.ylim(-10, 15)
plt.xlim(-10, 8)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(["Class 0", "Class 1", "Class 2", "Line class 0", "Line class 1", "Line class 2"], loc=(1.01, 0.3))


# 中心の三角形をどのように分類するか
mglearn.plots.plot_2d_classification(linear_svm, X, fill=True, alpha=.7)
mglearn.discrete_scatter(X[:,0], X[:,1],y)
line = np.linspace(-15, 15)
for coef, intercept, color in zip (linear_svm.coef_, linear_svm.intercept_, ['b', 'r', 'g']):
    plt.plot(line, -(line * coef[0] + intercept)/coef[1], c=color)
plt.legend(["Class 0", "Class 1", "Class 2", "Line class 0", "Line class 1", "Line class 2"], loc=(1.01, 0.3))
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")




