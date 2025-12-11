---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region editable=true slideshow={"slide_type": "slide"} -->
# Лекция 15: Методы понижения размерностей

МГТУ им. Н.Э. Баумана

Красников Александр Сергеевич

https://github.com/askras/mlda/

2024-2025
<!-- #endregion -->

```python editable=true slideshow={"slide_type": "skip"}
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
```

<!-- #region editable=true slideshow={"slide_type": "slide"} -->
Понижение размерности можно использовать для следующих целей:

* Сокращение ресурсоемкости алгоритмов
* Ослабление влияния проклятия размерности и тем самым уменьшение переобучения
* Переход к более информативным признакам

Сейчас мы будем понижать размерность ориентируясь как раз на эти цели.
Тогда этот процесс также можно называть и выделением признаков.
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "slide"} -->
## Отбор признаков

Cократить количество исходных признаков можно несколькими способами.
<!-- #endregion -->

```python editable=true slideshow={"slide_type": "slide"}
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

ds = fetch_california_housing()
X_, y = ds.data, ds.target
```

<!-- #region editable=true slideshow={"slide_type": "slide"} -->
### Отбор признаков на основе корреляции с целевой переменной
<!-- #endregion -->

```python editable=true slideshow={"slide_type": "slide"}
# Добавим "мешающий" признак
X = np.zeros((X_.shape[0],X_.shape[1]+1))
X[:,:-1] = X_
curr = np.random.randint(2, size=20640)
curr = np.array([elem if elem > 0 else elem-1 for elem in curr])
X[:,-1] = X[:,0]*curr
print(X.shape)

indices = np.arange(len(y))
np.random.shuffle(indices)
X = X[indices, :]
y = y[indices]

features_ind = np.arange(X.shape[1])
corrs = np.abs([np.corrcoef(X[:, i], y)[0][1] for i in features_ind])
importances_sort = np.argsort(corrs)
plt.barh(features_ind, corrs[importances_sort])
X = X[:, importances_sort]
```

```python editable=true slideshow={"slide_type": "slide"}
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

features_counts = np.arange(1, X.shape[1] + 1)

def scores_by_features_count(reg):
    scores = []
    for features_part in features_counts:
        X_part = X[:, :features_part]
        scores.append(cross_val_score(reg, X_part, y, cv=3).mean())
    return scores

plt.figure()
linreg_scores = scores_by_features_count(LinearRegression())
plt.plot(features_counts, linreg_scores, label='LinearRegression')

rf_scores = scores_by_features_count(RandomForestRegressor(n_estimators=100, max_depth=3))
plt.plot(features_counts, rf_scores, label='RandomForest')
plt.legend(loc='best');
```

<!-- #region editable=true slideshow={"slide_type": "slide"} -->
### Отбор признаков с помощью метода **SelectKBest**. 

Метод оставляет k признаков с самыми большими значениями некоторой статистики, которую используем для отбора. Приведем пример, в качестве статистики использующий совместную информацию признаков. Для признаков X и Y она задается следующей формулой:

$$I(X;Y)=\sum _{y\in Y}\sum _{x\in X}p(x,y)\log {\left({\frac {p(x,y)}{p(x)\,p(y)}}\right)}$$
<!-- #endregion -->

```python editable=true slideshow={"slide_type": "slide"}
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_regression

def scores_by_kbest_count(reg):
    scores = []
    for features_part in features_counts:
        X_new = SelectKBest(mutual_info_regression, k=features_part).fit_transform(X, y)
        scores.append(cross_val_score(reg, X_new, y, cv=3).mean())
    return scores

plt.figure()
linreg_scores2 = scores_by_kbest_count(LinearRegression())
plt.plot(features_counts, linreg_scores2, label='LinearRegression')

rf_scores2 = scores_by_kbest_count(RandomForestRegressor(n_estimators=100, max_depth=3))
plt.plot(features_counts, rf_scores2, label='RandomForest')
plt.legend(loc='best');
```

<!-- #region editable=true slideshow={"slide_type": "slide"} -->
### Рекурсивный отбор признаков

Выбираем алгоритм (estimator), применяем его, и он в результате своей работы присваивает веса всем признакам. Затем откидываем наименее важные признаки и снова запускаем estimator и т.д., до тех пор, пока не останется заранее заданное число признаков.
<!-- #endregion -->

```python editable=true slideshow={"slide_type": "slide"}
from sklearn.feature_selection import RFE
from sklearn.svm import SVC

def scores_by_rfe_count(reg):
    scores = []
    for features_part in features_counts:
        est = RandomForestRegressor(n_estimators=10, max_depth=3)
        X_rfe = RFE(estimator=est, n_features_to_select=features_part, step=1).fit_transform(X, y)
        scores.append(cross_val_score(reg, X_rfe, y, cv=3).mean())
    return scores

plt.figure()
linreg_scores3 = scores_by_rfe_count(LinearRegression())
plt.plot(features_counts, linreg_scores3, label='LinearRegression')

rf_scores3 = scores_by_rfe_count(RandomForestRegressor(n_estimators=100, max_depth=3))
plt.plot(features_counts, rf_scores3, label='RandomForest')
plt.legend(loc='best');
```

<!-- #region editable=true slideshow={"slide_type": "slide"} -->
Последние два метода при использовании RandomForestRegressor позволяют оставить довольно мало признаков, что может существенно ускорить модель.
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "slide"} -->
## Метод главных компонент (Principal Component Analysis, PCA)

Выделение новых признаков путем их отбора часто дает плохие результаты, и
в некоторых ситуациях такой подход практически бесполезен. Например, если
мы работаем с изображениями, у которых признаками являются яркости пикселей,
невозможно выбрать небольшой поднабор пикселей, который дает хорошую информацию о
содержимом картинки. 

Поэтому признаки нужно как-то комбинировать. Рассмотрим метод главных компонент.
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "slide"} -->
### Принципы PCA

PCA делает два важных упрощения задачи

1. **Игнорируется целевая переменная**
Это на первый взгляд кажется довольно странным, но на практике обычно не является
таким уж плохим. Это связано с тем, что часто данные устроены так, что имеют какую-то
внутреннюю структуру в пространстве меньшей размерности, которая никак не связана с
целевой переменной. Поэтому и оптимальные признаки можно строить не глядя на ответ.
2. **Строится линейная комбинация признаков**
Может сильно упростить задачу.
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "slide"} -->
### Теория

Обозначим $X$ - матрица объекты-признаки, с нулевым средним каждого признака,
а $w$ - некоторый единичный вектор. Тогда
$Xw$ задает величину проекций всех объектов на этот вектор. Далее ищется вектор,
который дает наибольшую дисперсию полученных проекций (то есть наибольшую дисперсию
вдоль этого направления):

$$
    \max_{w: \|w\|=1} \| Xw \|^2 =  \max_{w: \|w\|=1} w^T X^T X w
$$

Подходящий вектор тогда равен собственному вектору матрицы $X^T X$ с наибольшим собственным
значением. После этого все пространство проецируется на ортогональное дополнение к вектору
$w$ и процесс повторяется.
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "slide"} -->
### PCA на плоскости

Для начала посмотрим на метод PCA на плоскости для того, чтобы лучше понять, как он устроен. Попробуем специально сделать один из признаков более значимым и проверим, что PCA это обнаружит. Сгенерируем выборку из двухмерного нормального распределения с нулевым математическим ожиданием. 
<!-- #endregion -->

```python editable=true slideshow={"slide_type": "fragment"}
np.random.seed(314512)

data_synth_1 = np.random.multivariate_normal(
    mean=[0, 0], 
    cov=[[4, 0], 
         [0, 1]],
    size=1000
)
```

<!-- #region editable=true slideshow={"slide_type": "slide"} -->
Теперь изобразим точки выборки на плоскости и применим к ним PCA для нахождения главных компонент. В результате работы PCA из sklearn в `dec.components_` будут лежать главные направления (нормированные), а в `dec.explained_variance_` &mdash; дисперсия, которую объясняет каждая компонента. Изобразим на нашем графике эти направления, умножив их на дисперсию для наглядного отображения их значимости.
<!-- #endregion -->

```python editable=true slideshow={"slide_type": "slide"}
from sklearn.decomposition import PCA

def PCA_show(dataset):
    plt.scatter(*zip(*dataset), alpha=0.5)
    
    dec = PCA()
    dec.fit(dataset)
    ax = plt.gca()
    for comp_ind in range(dec.components_.shape[0]):
        component = dec.components_[comp_ind, :]
        var = dec.explained_variance_[comp_ind]
        start, end = dec.mean_, component * var
        ax.arrow(start[0], start[1], end[0], end[1],
                 head_width=0.2, head_length=0.4, fc='r', ec='r')
    
    ax.set_aspect('equal', adjustable='box')

plt.figure(figsize=(16, 8))
PCA_show(data_synth_1)
```

<!-- #region editable=true slideshow={"slide_type": "slide"} -->
Видим, что PCA все правильно нашел. Но это, конечно, можно было сделать и просто посчитав
дисперсию каждого признака. Повернем наши данные на некоторый фиксированный угол и проверим,
что для PCA это ничего не изменит.
<!-- #endregion -->

```python editable=true slideshow={"slide_type": "fragment"}
angle = np.pi / 6
rotate = np.array([
        [np.cos(angle), - np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])
data_synth_2 = rotate.dot(data_synth_1.T).T

plt.figure(figsize=(16, 8))
PCA_show(data_synth_2)
```

<!-- #region editable=true slideshow={"slide_type": "slide"} -->
Ниже пара примеров, где PCA отработал не так хорошо (в том смысле, что направления задают не очень хорошие признаки).
<!-- #endregion -->

```python editable=true slideshow={"slide_type": "fragment"}
from sklearn.datasets import make_circles, make_moons, make_blobs

np.random.seed(54242)

data_synth_bad = [
    make_circles(n_samples=1000, factor=0.2, noise=0.1)[0]*2,
    make_moons(n_samples=1000, noise=0.1)[0]*2,
    make_blobs(n_samples=1000, n_features=2, centers=4)[0]/5,
    np.random.multivariate_normal(
        mean=[0, 1.5], 
        cov=[[3, 1], 
             [1, 1]],
        size=1000),
]

plt.figure(figsize=(16,8))
rows, cols = 2, 2
for i, data in enumerate(data_synth_bad):
    plt.subplot(rows, cols, i + 1)
    PCA_show(data)
```

<!-- #region editable=true slideshow={"slide_type": "slide"} -->
## Пример. Лица людей

Рассмотрим датасет с фотографиями лиц людей и применим к его признакам PCA. Ниже изображены примеры лиц из базы, и последняя картинка &mdash; это "среднее лицо".
<!-- #endregion -->

```python editable=true slideshow={"slide_type": "fragment"}
from sklearn.datasets import fetch_olivetti_faces

faces = fetch_olivetti_faces(shuffle=True, random_state=432542)
faces_images = faces.data
faces_ids = faces.target
image_shape = (64, 64)
    
mean_face = faces_images.mean(axis=0)

plt.figure(figsize=(16, 8))
rows, cols = 2, 4
n_samples = rows * cols
for i in range(n_samples - 1):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(faces_images[i, :].reshape(image_shape), interpolation='none',
               cmap='gray')
    plt.xticks(())
    plt.yticks(())
    
plt.subplot(rows, cols, n_samples)
plt.imshow(mean_face.reshape(image_shape), interpolation='none',
           cmap='gray')
plt.xticks(())
_ = plt.yticks(())
```

<!-- #region editable=true slideshow={"slide_type": "fragment"} -->
Теперь найдем главные компоненты
<!-- #endregion -->

```python editable=true slideshow={"slide_type": "fragment"}
model_pca = PCA()
faces_images -= mean_face  # отнормировали данные к нулевому среднему
model_pca.fit(faces_images)

plt.figure(figsize=(16, 8))
rows, cols = 2, 4
n_samples = rows * cols
for i in range(n_samples):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(model_pca.components_[i, :].reshape(image_shape), interpolation='none', cmap='gray')
    plt.xticks(())
    plt.yticks(())
```

<!-- #region editable=true slideshow={"slide_type": "slide"} -->
Получилось жутковато, что уже неплохо, но есть ли от этого какая-то польза?
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "slide"} -->
#### 1. новые признаки дают более высокое качество классификации.
<!-- #endregion -->

```python editable=true slideshow={"slide_type": "fragment"}
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

gscv_rf = GridSearchCV(
    RandomForestClassifier(),
    {'n_estimators': [100, 200, 500, 800], 'max_depth': [2, 3, 4, 5]}, cv=5
)
```

```python editable=true slideshow={"slide_type": "slide"}
%%time
gscv_rf.fit(faces_images, faces_ids)
print('Точность на исходных данных:', gscv_rf.best_score_)
```

```python editable=true slideshow={"slide_type": "slide"}
%%time
gscv_rf.fit(model_pca.transform(faces_images)[:,:100], faces_ids)
print('Точность на преобразованных данных:',gscv_rf.best_score_)
```

<!-- #region editable=true slideshow={"slide_type": "slide"} -->
На практике можно выбирать столько главных компонент, чтобы оставить $90\%$ дисперсии исходных данных. В данном случае для этого достаточно выделить около $60$ главных компонент, то есть снизить размерность с $4096$ признаков до $60$.
<!-- #endregion -->

```python editable=true slideshow={"slide_type": "fragment"}
faces_images.shape
```

```python editable=true slideshow={"slide_type": "fragment"}
plt.figure(figsize=(10,7))
plt.plot(np.cumsum(model_pca.explained_variance_ratio_), color='k', lw=2)
plt.xlabel('Количество компонент')
plt.ylabel('Общая объясненная разница')
plt.xlim(0, 63)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.axhline(0.9, c='r')
plt.show();
```

<!-- #region editable=true slideshow={"slide_type": "slide"} -->
#### 2. Можно использовать для компактного хранения данных.

Для этого объекты трансформируются в новое пространство, и из него выкидываются самые незначимые признаки. Ниже приведены результаты сжатия в 20 и  50 раз.
<!-- #endregion -->

```python editable=true slideshow={"slide_type": "fragment"}
# Сжатие изображений 

base_size = image_shape[0] * image_shape[1]

def compress_and_show(compress_ratio):
    model_pca = PCA(n_components=int(base_size * compress_ratio))
    model_pca.fit(faces_images)

    faces_compressed = model_pca.transform(faces_images)
    
    # обратное преобразование
    faces_restored = model_pca.inverse_transform(faces_compressed) + mean_face

    plt.figure(figsize=(16, 8))
    rows, cols = 2, 4
    n_samples = rows * cols
    for i in range(n_samples):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(faces_restored[i, :].reshape(image_shape), interpolation='none',
                   cmap='gray')
        plt.xticks(())
        plt.yticks(())
```

```python editable=true slideshow={"slide_type": "slide"}
print('Исходные изображения')
mean_face = faces_images.mean(axis=0)

plt.figure(figsize=(16, 8))
rows, cols = 2, 4
n_samples = rows * cols
for i in range(n_samples - 1):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(faces_images[i, :].reshape(image_shape), interpolation='none',
               cmap='gray')
    plt.xticks(())
    plt.yticks(())

plt.subplot(rows, cols, n_samples)
plt.imshow(mean_face.reshape(image_shape), interpolation='none',
           cmap='gray')
plt.xticks(())
_ = plt.yticks(())

print('Сжатие в 20 раз')
compress_and_show(0.05)

print('Сжатие в 50 раз')
compress_and_show(0.02)
```

<!-- #region editable=true slideshow={"slide_type": "slide"} -->
## PCA с ядрами

Так как PCA фактически работает не исходными признаками, а с матрицей их ковариаций, можно использовать для ее вычисления вместо скалярного произведения $\langle x_i, x_j \rangle$ произвольное ядро $K(x_i, x_j)$. Это будет соответствовать переходу в другое пространство. Единственная проблема &mdash; непонятно, как подбирать ядро.

Ниже приведены примеры объектов в исходном пространстве (похожие группы обозначены одним цветом для наглядности), и результат их трансформации в новые пространства (для разных ядер). Если результаты получаются линейно разделимыми &mdash; значит мы выбрали подходящее ядро.
<!-- #endregion -->

```python editable=true slideshow={"slide_type": "slide"}
from sklearn.decomposition import KernelPCA

def KPCA_show(X, y):
    reds = y == 0
    blues = y == 1
    
    plt.figure(figsize=(8, 8))
    rows, cols = 2, 2
    plt.subplot(rows, cols, 1)
    plt.scatter(X[reds, 0], X[reds, 1], alpha=0.5, c='r')
    plt.scatter(X[blues, 0], X[blues, 1], alpha=0.5, c='b')
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    
    kernels_params = [
        dict(kernel='rbf', gamma=10),
        dict(kernel='poly', gamma=10),
        dict(kernel='cosine', gamma=10),
    ]
    
    for i, p in enumerate(kernels_params):
        dec = KernelPCA(**p)
        X_transformed = dec.fit_transform(X)
        
        plt.subplot(rows, cols, i + 2)
        plt.scatter(X_transformed[reds, 0], X_transformed[reds, 1], alpha=0.5, c='r')
        plt.scatter(X_transformed[blues, 0], X_transformed[blues, 1], alpha=0.5, c='b')
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
```

```python editable=true slideshow={"slide_type": "slide"}
np.random.seed(54242)
KPCA_show(*make_circles(n_samples=1000, factor=0.2, noise=0.1))
```

```python editable=true slideshow={"slide_type": "slide"}
np.random.seed(54242)
KPCA_show(*make_moons(n_samples=1000, noise=0.1))
```

<!-- #region editable=true slideshow={"slide_type": "slide"} -->
## TSNE (t-distributed Stohastic Neighbor Embedding)

Джефри Хинтон в 2008 году, придумал[новый методв изуализации данных.](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) Основная идея метода состоит в поиске отображения из многомерного признакового пространства на плоскость (или в 3D, но почти всегда выбирают 2D), чтоб точки, которые были далеко друг от друга, на плоскости тоже оказались удаленными, а близкие точки – также отобразились на близкие. То есть neighbor embedding – это своего рода поиск нового представления данных, при котором сохраняется соседство.

Попробуем взять данные о рукописных цифрах и визуализируем их с помощью PCA. 
<!-- #endregion -->

```python editable=true slideshow={"slide_type": "slide"}
from sklearn.datasets import load_digits
digits = load_digits()

X = digits.data
y = digits.target

plt.figure(figsize=(16, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X[i,:].reshape([8,8]), cmap='gray');
```

```python editable=true slideshow={"slide_type": "slide"}
X.shape
```

<!-- #region editable=true slideshow={"slide_type": "fragment"} -->
Получается, размерность признакового пространства здесь – 64. Но давайте снизим размерность всего до 2 и увидим, что даже на глаз рукописные цифры неплохо разделяются на кластеры.
<!-- #endregion -->

```python editable=true slideshow={"slide_type": "slide"}
model_pca = PCA(n_components=2)
X_reduced = model_pca.fit_transform(X)

plt.figure(figsize=(12,10))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, 
            edgecolor='none', alpha=0.7, s=40,
            cmap=plt.get_cmap('nipy_spectral', 10))
plt.colorbar()
plt.title('MNIST. PCA проекция');
```

<!-- #region editable=true slideshow={"slide_type": "slide"} -->
Попробуем сделать то же самое с помощью t-SNE. Картинка получится лучше, так как у PCA есть существенное ограничение - он находит только линейные комбинации исходных признаков (если не добавить какое-нибудь ядро). 
<!-- #endregion -->

```python editable=true slideshow={"slide_type": "fragment"}
from sklearn.manifold import TSNE

tsne = TSNE(n_jobs=4, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(12,10))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, 
            edgecolor='none', alpha=0.7, s=40,
            cmap=plt.get_cmap('nipy_spectral', 10))

plt.colorbar()
plt.title('MNIST. t-SNE проекция');
```

<!-- #region editable=true slideshow={"slide_type": "slide"} -->
У метода есть параметр `Perplexity`, который отвечает за то, насколько сильно точки могут разлететься друг от друга. 
<!-- #endregion -->

```python editable=true slideshow={"slide_type": "fragment"}
tsne = TSNE(n_jobs=4, perplexity=2, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(12,10))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, 
            edgecolor='none', alpha=0.7, s=40,
            cmap=plt.get_cmap('nipy_spectral', 10))

plt.colorbar()
plt.title('MNIST. t-SNE проекция');
```

<!-- #region editable=true slideshow={"slide_type": ""} -->

<!-- #endregion -->
