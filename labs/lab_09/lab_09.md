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

<!-- #region id="UqqA7Jph8T2o" -->
## Деревья решений

Продолжительность работы: - 4 часа.

Мягкий дедлайн (10 баллов): 11.12.2025

Жесткий дедлайн (5 баллов): 18.12.2025
<!-- #endregion -->

<!-- #region id="9iAZwRjCeAQO" -->
В данной домашней работе требуется реализовать разбиение элементов выборки в вершине дерева.
<!-- #endregion -->

<!-- #region id="7zl0dnOpd3xX" -->
$R$ - множество объектов в разбиваемой вершине, $j$ - номер признака, по которому происходит разбиение, $t$ - порог разбиения.

Критерий ошибки:

$$
Q(R, j, t) = \frac{|R_\ell|}{|R|}H(R_\ell) + \frac{|R_r|}{|R_m|}H(R_r) \to \min_{j, t}
$$

$R_\ell$ - множество объектов в левом поддереве, $R_r$ - множество объектов в правом поддереве.

$H(R)$ - критерий информативности, с помощью которого можно оценить качество распределения целевой переменной среди объектов множества $R$.
<!-- #endregion -->

```python executionInfo={"elapsed": 2409, "status": "ok", "timestamp": 1696944917459, "user": {"displayName": "Sergey Korpachev", "userId": "09181340988160569540"}, "user_tz": -180} id="kHeOJOhD-TVX"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Iterable, List, Tuple

from sklearn.model_selection import train_test_split
```

```python executionInfo={"elapsed": 5, "status": "ok", "timestamp": 1696944604231, "user": {"displayName": "Sergey Korpachev", "userId": "09181340988160569540"}, "user_tz": -180} id="D_kJdSOIKL2Z"
# Загрузим набор данных по недвижимости в Бостоне
data = pd.read_csv('boston_house_prices.csv')
data.head()
```

<!-- #region id="b1KE7WIFKp51" -->
**Описание набора данных:**  

Boston Housing содержит данные, собранные Службой переписи населения США (англ. U.S Census Service), касающиеся недвижимости в районах Бостона. Набор данных состоит из 13 признаков и 506 строк и также предоставляет такую информацию, как уровень преступности (CRIM), ставка налога на недвижимость (TAX), возраст людей, которым принадлежит дом (AGE), соотношение числа учащихся и преподавателей в районе (PTRATIO) и другие. Данный набор данных используется для предсказания следующих целевых переменных: средняя стоимость дома (MEDV) и уровень закиси азота (NOX).

Описание набора данных можно посмотреть здесь: https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html
<!-- #endregion -->

```python executionInfo={"elapsed": 407, "status": "ok", "timestamp": 1696945050297, "user": {"displayName": "Sergey Korpachev", "userId": "09181340988160569540"}, "user_tz": -180} id="3KLql08wLMKy"
feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

X = pd.DataFrame(data, columns=feature_names, index=range(len(data)))
y = pd.DataFrame(data, columns=['MEDV'], index=range(len(data)))

X['target'] = y
```

```python executionInfo={"elapsed": 422, "status": "ok", "timestamp": 1696945067442, "user": {"displayName": "Sergey Korpachev", "userId": "09181340988160569540"}, "user_tz": -180} id="QMs9TfkuSme0"
X_train, X_test = train_test_split(X, test_size=0.25, random_state=13)
```

<!-- #region id="hirWI1Vw4Fle" -->
**Задание 1**: 
Реализуйте подсчет критерия ошибки. Для этого реализуйте функции для подсчета значения критерия информативности, а также для разбиения вершины.
<!-- #endregion -->

```python executionInfo={"elapsed": 531, "status": "ok", "timestamp": 1696945074754, "user": {"displayName": "Sergey Korpachev", "userId": "09181340988160569540"}, "user_tz": -180} id="5bGstPZ14Flf"
def H(R: np.array) -> float:
    """Вычислить критерий информативности для фиксированного набора объектов R.
    Предполагается, что последний столбец содержит целевое значение
    """
    # Здесь должен быть ваш код


def split_node(R: np.array, feature: str, t: float) -> Iterable[np.array]:
    """
    Разделить фиксированный набор объектов R по признаку feature с пороговым значением t
    """
    # Здесь должен быть ваш код


def Q(R: np.array, feature: str, t: float) -> float:
    """
    Вычислить функционал качества для заданных параметров разделения
    """
    # Здесь должен быть ваш код

```

<!-- #region id="z5vMn7Yu4Flg" -->
**Задание 2**:
Переберите все возможные разбиения обучающей выборки по одному из признаков и постройте график критерия ошибки в зависимости от значения порога.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 493} executionInfo={"elapsed": 1006, "status": "ok", "timestamp": 1696945092152, "user": {"displayName": "Sergey Korpachev", "userId": "09181340988160569540"}, "user_tz": -180} id="73d0n-Ht4Flh" outputId="0f7492ef-479d-4f66-ac07-e5bd30418269"
# Здесь должен быть ваш код
```

<!-- #region id="cdNVqLH24Flj" -->
**Задание 3**:
Напишите функцию, находящую оптимальное разбиение данной вершины по данному признаку.
<!-- #endregion -->

```python executionInfo={"elapsed": 428, "status": "ok", "timestamp": 1696945125413, "user": {"displayName": "Sergey Korpachev", "userId": "09181340988160569540"}, "user_tz": -180} id="JnK6p2FU4Flk"
def get_optimal_split(R: np.array, feature: str) -> Tuple[float, List[float]]:
    # Здесь должен быть ваш код
```

<!-- #region id="WTwCYIgc4Fll" -->
**Задание 4**: 
Для первого разбиения найдите признак, показывающий наилучшее качество. Каков порог разбиения и значение качества? Постройте график критерия ошибки для данного признака в зависимости от значения порога.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3139, "status": "ok", "timestamp": 1696945163220, "user": {"displayName": "Sergey Korpachev", "userId": "09181340988160569540"}, "user_tz": -180} id="GUl5daTc4Flo" outputId="23455857-fab8-472f-c760-504b732e147e"
# Здесь должен быть ваш код
```

<!-- #region id="qaSseANG4Flq" -->
**Задание 5**:
 Изобразите разбиение визуально. Для этого постройте диаграмму рассеяния целевой переменной в зависимости от значения входного признака. Далее изобразите вертикальную линию, соответствующую порогу разбиения.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 563} executionInfo={"elapsed": 432, "status": "ok", "timestamp": 1696945203129, "user": {"displayName": "Sergey Korpachev", "userId": "09181340988160569540"}, "user_tz": -180} id="Cjw3cznv8Qn2" outputId="47553417-1bdc-43c5-f6f3-e932dba18b45"
# Здесь должен быть ваш код
```

**Задание 5**:
Постройте модель обучения с помощью библиотеки модуля [Decision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html).

```python
from sklearn.tree import DecisionTreeRegressor
# Здесь должен быть ваш код
```
