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

for n_neighbors in neighbors_setting:
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






