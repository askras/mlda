---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region id="rZOA7MoXtA-e" -->
# Кластеризация

**Цель работы:** получение практических навыков использования методов кластеризации.

Продолжительность работы: - 4 часа.

Мягкий дедлайн (10 баллов): 04.12.2025

Жесткий дедлайн (5 баллов): 11.12.2025
<!-- #endregion -->

<!-- #region id="rZOA7MoXtA-e" -->
## Задание 1. Кластеризация исполнителей по жанрам
<!-- #endregion -->

<!-- #region id="kabxpe1rpmPc" -->
В этом задании необходимо кластеризовать исполнителей по жанрам на основе данных о прослушивании.

В файле music_listening.csv по строкам стоят пользователи, а по столбцам - исполнители.

Для каждой пары (пользователь,исполнитель) в таблице стоит число - доля (процент) прослушивания этого исполнителя выбранным пользователем.
<!-- #endregion -->

<!-- #region id="wF3gCvLCtYyv" -->
### Импорт библиотек, загрузка данных
<!-- #endregion -->

```python id="uprlaM05NGvI"
import pandas as pd
```

```python id="BeE8Io3_uhi3"
ratings = pd.read_csv('music_listening.csv', low_memory=False)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 297} id="ZD0jf_i1Oy5d" outputId="e4f54839-20dc-4c19-a671-e1be5490fad6"
ratings.head()
```

<!-- #region id="pHTPXIEvp_9f" -->
Транспонируем матрицу ratings, чтобы по строкам стояли исполнители.
<!-- #endregion -->

```python id="BfKDPzjoPxKE"
ratings = ratings.T
ratings.head()
```

<!-- #region id="B9tfrtCoqFbQ" -->
Удалите строку под названием `user`.
<!-- #endregion -->

```python id="FNMhITClOeyN"
# Здесь должен быть ваш код
...
```

<!-- #region id="eKZfzlnMBO-M" -->
Сколько строк осталось в матрице ratings?
<!-- #endregion -->

```python id="eKZfzlnMBO-M"
# Здесь должен быть ваш код
...
```

<!-- #region id="rxxnjpNVqJYN" -->
Заполните пропуски нулями.
<!-- #endregion -->

```python id="iKduzcA2OiFV"
# Здесь должен быть ваш код
...
```

<!-- #region id="j3ca2KiJqL9J" -->
Нормализуйте данные при помощи `normalize`.
<!-- #endregion -->

```python id="CVxxov5dqSb2"
from sklearn.preprocessing import normalize

# Здесь должен быть ваш код
...
```

<!-- #region id="Ne6bESslqUXp" -->
Примените KMeans с 5ю кластерами на преобразованной матрице (сделайте fit, а затем вычислите кластеры при помощи predict).
<!-- #endregion -->

```python id="cP13pV-dNo5s"
from sklearn.cluster import KMeans

# Здесь должен быть ваш код
...
```

<!-- #region id="uZk0MMMUqiYM" -->
Выведите на экран центры кластеров (центроиды)
<!-- #endregion -->

```python id="6mBQ-C1Bqmm3"
centroids = ... # Здесь должен быть ваш код
```

<!-- #region id="9eFLNiA_CRFq" -->
Для каждого кластера найдем топ-10 исполнителей, наиболее близких к центроидам соотвествующего кластера.

Схожесть исполнителей будем считать по косинусной мере (spatial.distance.cosine).
<!-- #endregion -->

<!-- #region id="9eFLNiA_CRFq" -->
Вычислите расстояние между "the beatles" и "coldplay". Ответ округлите до сотых.
<!-- #endregion -->

```python id="oYtgZwSOCiHG"
from scipy import spatial

# Здесь должен быть ваш код
...
```

<!-- #region id="BM3k07IRqnhF" -->
Ниже приведена функция, принимающая на вход:
* np.array points - все точки кластера
* pt - центроид кластера
* K = 10 - число
Функция возвращает K индексов объектов (строк в массиве points), ближайших к центроиду.
<!-- #endregion -->

```python id="8f-Pm9pNBYW2"
def pClosest(points, pt, K=10):
    ind = [i[0] for i in sorted(enumerate(points), key=lambda x: spatial.distance.cosine(x[1], pt))]
    return ind[:K]
```

<!-- #region id="JPE-V0c9B_AC" -->
Примените функцию pClosest (или придумайте свой подход) и выведите для каждого кластера названия топ-10 исполнителей, ближайших к центроиду.
<!-- #endregion -->

```python id="PX1NO6CJqutV"
# Здесь должен быть ваш код
...
```

<!-- #region id="fk8TRHInqv_j" -->
Проинтерпретируйте результат. Что можно сказать о смысле кластеров?
<!-- #endregion -->

<!-- #region id="INk3CCWLa-ZJ" -->
## Задание 2. Сравнение алгоритмов кластеризации

Попробуйте разные методы для поиска кластеров.
Не забудьте измерить время работы работы каждого метода.
<!-- #endregion -->

```python id="4fgkK-YubAzW" colab={"base_uri": "https://localhost:8080/", "height": 447} outputId="defcf8f5-78f8-44d5-ec81-900e7621d79f"
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons

data = make_moons(n_samples=100, noise=0.1, random_state=42)

X = data[0]
y = data[1]

plt.scatter(X[:,0], X[:,1], c=y);
```

<!-- #region id="E8Aj-5rMbv19" -->
### KMeans.

Попробуйте найти кластеры при помощи KMeans
<!-- #endregion -->

```python id="rcZyFe8KbPze"
from sklearn.cluster import KMeans

# Здесь должен быть ваш код
...
```

<!-- #region id="KQCV8we8b1OD" -->
### DBSCAN

Подберите $\varepsilon$ и min_samples в DBSCAN, чтобы наилучшим образом найти кластеры.
Ищите гиперпараметры из диапазонов:
* eps in [0.05, 0.1, 0.2, 0.28, 0.3, 0.32]
* min_samples in [4, 5, 6, 7]
<!-- #endregion -->

```python id="XVFoaJqtb6z9"
from sklearn.cluster import DBSCAN

# Здесь должен быть ваш код
...
```

<!-- #region id="gVT2XXpXb7iY" -->
### Иерархическая кластеризация

Используйте иерархическую кластеризацию для поиска кластеров.
Задайте в методе 2 кластера.
Подберите гиперпараметр linkage из списка ['ward', 'complete', 'average', 'single'], дающий наилучший результат.
<!-- #endregion -->

```python id="o1K8jDR7b_BW"
from sclearn.cluster import AgglomerativeClustering

# Здесь должен быть ваш код
...
```

<!-- #region id="JpBvTJi1b_16" -->
## Спектральная кластеризация

Попробуйте найти кластеры при помощи спектральной кластеризации.
Задайте 2 кластера, affinity='nearest_neighbors'.
Подберите гиперпараметр n_neighbors из диапазона [1,2,...,19], чтобы добиться наилучшего результата.
<!-- #endregion -->

```python id="XqAacOmXcCPv"
from sklearn.cluster import SpectralClustering

# Здесь должен быть ваш код
...
```

<!-- #region id="7e-_7M4PcID8" -->
## Выводы

Сделайте выводы: какой метод сработал лучше других? какой метод сработал быстрее? есть ли метод, наилучший и по качеству, и по времени одновременно?
<!-- #endregion -->
