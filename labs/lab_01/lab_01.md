---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.3
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region id="7247b8d4" editable=true slideshow={"slide_type": ""} -->
# Лабораторная работа 1. Пакет NumPy


Продолжительность работы: - 4 часа.

Мягкий дедлайн (10 баллов): 25.09.2026

Жесткий дедлайн (5 баллов): 02.10.2026

Оценка за каждое задание указана в комментариях перед заданием.

Задание выполняется самостоятельно, в противном случае все причастные получат 0 баллов :)
Если вы нашли решение любого из заданий (или его части) в открытом источнике, необходимо указать ссылку на этот источник в отдельном блоке в конце своей работы. 
В противном случае **работа также будет оценена в 0 баллов**.
<!-- #endregion -->

<!-- #region id="1966e3d0" editable=true slideshow={"slide_type": ""} -->
При выполнении заданий запрещено использовать **while**, **for**, **if**. 
Все операции должны выполняться с помощью numpy. 
Напомним, что использование, например, max вместо np.max также является неоптимальным шагом.
Решение будет засчитано, если оно удовлетворяет условиям выше и проходит asserts.
<!-- #endregion -->

```python executionInfo={"elapsed": 1081, "status": "ok", "timestamp": 1694439757773, "user": {"displayName": "Sergey Korpachev", "userId": "09181340988160569540"}, "user_tz": -180} id="03cf459c" editable=true slideshow={"slide_type": ""}
import numpy as np

score = 0
```

<!-- #region id="cDsKeK4EaWrE" editable=true slideshow={"slide_type": ""} -->
## Задание 1 (1 балл)
<!-- #endregion -->

```python executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1694439759790, "user": {"displayName": "Sergey Korpachev", "userId": "09181340988160569540"}, "user_tz": -180} id="439425f7" editable=true slideshow={"slide_type": ""}
# задание 1 (1 балл)

def max_after_zero(x: np.array) -> int:
    """
    Задание: найти максимальный элемент массива среди элементов, которым предшествует ноль
      
    Вход: np.array([0, 2, 0, 3])
    Выход: 3
    """
    assert False, 'Не реализовано!' # Здесь должен быть ваш код
```

```python colab={"base_uri": "https://localhost:8080/", "height": 293} executionInfo={"elapsed": 252, "status": "error", "timestamp": 1694439761443, "user": {"displayName": "Sergey Korpachev", "userId": "09181340988160569540"}, "user_tz": -180} id="a58b05aa" outputId="bfa9232f-9fe7-4333-da96-0a3cc4f0f59d" editable=false slideshow={"slide_type": ""}
%%capture

x = np.array([0, 1, 12, 0, 6, 0, 10, 0])
assert max_after_zero(x) == 10, 'Тест не пройден'

x = np.array([0, 3, 2, 0, 8, 0, 1, 10])
assert max_after_zero(x) == 8, 'Тест не пройден'

print("Выполнено")
score += 1
```

<!-- #region id="zT54XZpBaeHX" editable=true slideshow={"slide_type": ""} -->
## Задание 2 (1 балл)
<!-- #endregion -->

```python executionInfo={"elapsed": 10, "status": "ok", "timestamp": 1694439791603, "user": {"displayName": "Sergey Korpachev", "userId": "09181340988160569540"}, "user_tz": -180} id="0a3ff5e4" editable=true slideshow={"slide_type": ""}
# задание 2 (1 балл)

def block_matrix(block: np.array) -> np.array:
    """
    Задание: построить блочную матрицу из четырех блоков, где каждый блок представляет собой заданную матрицу

    Вход: np.array([[1, 2], [3, 4]])
    Выход: np.array([[1, 2, 1, 2],
                     [3, 4, 3, 4],
                     [1, 2, 1, 2],
                     [3, 4, 3, 4]])
    """
    assert False, 'Не реализовано!' # Здесь должен быть ваш код
```

```python colab={"base_uri": "https://localhost:8080/", "height": 327} executionInfo={"elapsed": 9, "status": "error", "timestamp": 1694439791604, "user": {"displayName": "Sergey Korpachev", "userId": "09181340988160569540"}, "user_tz": -180} id="f0ce9850" outputId="c67f53b4-ecc6-4fa7-81ac-432927da7dde" editable=false slideshow={"slide_type": ""}
%%capture

block = np.array([[1, 3, 3], [7, 0, 0]])
assert np.allclose(
    block_matrix(block),
    np.array([[1, 3, 3, 1, 3, 3],
              [7, 0, 0, 7, 0, 0],
              [1, 3, 3, 1, 3, 3],
              [7, 0, 0, 7, 0, 0]])
), 'Тест не пройден'

print("Выполнено")
score += 1
```

<!-- #region id="dzvsvRm6apVb" editable=true slideshow={"slide_type": ""} -->
## Задание 3 (1 балл)
<!-- #endregion -->

```python executionInfo={"elapsed": 246, "status": "ok", "timestamp": 1694439796375, "user": {"displayName": "Sergey Korpachev", "userId": "09181340988160569540"}, "user_tz": -180} id="b4535fbf" editable=true slideshow={"slide_type": ""}
# задание 3 (1 балл)

def diag_prod(matrix: np.array) -> int:
    """
    Задание: вычислить произведение всех ненулевых диагональных элементов квадратной матрицы

    Вход: np.array([[3, 5, 1, 4],
                    [6, 2, 7, 9],
                    [3, 6, 0, 8],
                    [1, 3, 4, 6]])
    Выход: 36
    """
    assert False, 'Не реализовано!' # Здесь должен быть ваш код
```

```python colab={"base_uri": "https://localhost:8080/", "height": 310} executionInfo={"elapsed": 5, "status": "error", "timestamp": 1694439797817, "user": {"displayName": "Sergey Korpachev", "userId": "09181340988160569540"}, "user_tz": -180} id="fa039421" outputId="080cb3f5-b033-4cf2-bb9a-8f9c852090a6" editable=false slideshow={"slide_type": ""}
%%capture

matrix = np.array([[0, 1, 2, 3],
                   [4, 5, 6, 7],
                   [8, 9, 10, 11],
                   [12, 13, 14, 15]])
assert diag_prod(matrix) == 750, 'Тест не пройден'

print("Выполнено")
score += 1
```

<!-- #region id="1U5HMX3Marze" -->
### Задание 4 (1 балл)
<!-- #endregion -->

```python executionInfo={"elapsed": 5, "status": "ok", "timestamp": 1694439800023, "user": {"displayName": "Sergey Korpachev", "userId": "09181340988160569540"}, "user_tz": -180} id="cfa98502" editable=true slideshow={"slide_type": ""}
# задание 4 (1 балл)

from typing import Tuple

class StandardScaler:
    """
    Задание: класс реализует StandardScaler из библиотеки sklearn
    
    см. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    В качестве входных данных метод fit принимает матрицу, в которой признаки объектов расположены в столбцах 
    Метод fit должен вычислять среднее значение (mean_) и дисперсию (var_) для каждого из признаков (столбца), 
    и сохранять их в атрубутах объекта self.mean_ и self.var_ соответственно.    
    
    Метод transform должен нормализовать матрицу с помощью предварительно вычисленных mean_ и sigma, 
    где sigma = sqrt(var_) - среднеквадратическое отклонение

    Вход: np.array([[2, 603, 250], 
                    [1, 154, 500], 
                    [7, 893, 350]])
    Выход: np.array([[-0.50800051,  0.17433393, -1.13554995],
                     [-0.88900089, -1.3025705 ,  1.29777137],
                     [ 1.3970014 ,  1.12823657, -0.16222142]])
    """
        
    def fit(self, X: np.array) -> None:
        assert False, 'Не реализовано!' # Здесь должен быть ваш код

    def transform(self, X: np.array) -> np.array:
        assert False, 'Не реализовано!' # Здесь должен быть ваш код
```

```python colab={"base_uri": "https://localhost:8080/", "height": 361} executionInfo={"elapsed": 6, "status": "error", "timestamp": 1694439800521, "user": {"displayName": "Sergey Korpachev", "userId": "09181340988160569540"}, "user_tz": -180} id="352d0513" outputId="42f66f0a-e221-4c90-f081-2007cd20b811" editable=false slideshow={"slide_type": ""}
%%capture

matrix = np.array([[1, 4, 4200], [0, 10, 5000], [1, 2, 1000]])

scaler = StandardScaler()
scaler.fit(matrix)

assert np.allclose(
    scaler.mean_,
    np.array([0.66667, 5.3333, 3400])
), 'Тест не пройден. Некорректное значение scaler.mean_'

assert np.allclose(
    scaler.var_,
    np.array([0.22222, 11.5556, 2986666.67])
), 'Тест не пройден. Некорректное значение scaler.var_'

assert np.allclose(
    scaler.transform(matrix),
    np.array([[ 0.7071, -0.39223,  0.46291],
              [-1.4142,  1.37281,  0.92582],
              [ 0.7071, -0.98058, -1.38873]])
), 'Тест не пройден. Некорректный результат scaler.transform(matrix)'


print("Выполнено")
score += 1
```

<!-- #region id="VgfO8yt7atav" editable=true slideshow={"slide_type": ""} -->
### Задание 5 (1 балл)
<!-- #endregion -->

```python executionInfo={"elapsed": 3, "status": "ok", "timestamp": 1694439802963, "user": {"displayName": "Sergey Korpachev", "userId": "09181340988160569540"}, "user_tz": -180} id="7d68dc80" slideshow={"slide_type": ""}
# задание 5 (1 балл)

def antiderivative(coefs: np.array, const: float) -> np.array:
    """
    Задание: Вычислить первообразную полинома

    coefs - массив коэффициентов полинома
    const - произвольная постоянная
    Массив коэффициентов [6, 0, 1] соответствует 6x^2 + 0x^1 + 1
    Соответствующая первообразная будет иметь вид: 2x^3 + 0x^2 + 1x + const,
    В результате получается массив коэффициентов [2, 0, 1, const]
        
    Вход: [8, 12, 8, 1], 42
    Выход: [2., 4., 4., 1., 42.]
    """
    assert False, 'Не реализовано!' # Здесь должен быть ваш код
```

```python colab={"base_uri": "https://localhost:8080/", "height": 327} executionInfo={"elapsed": 6, "status": "error", "timestamp": 1694439803375, "user": {"displayName": "Sergey Korpachev", "userId": "09181340988160569540"}, "user_tz": -180} id="41288733" outputId="9e4b86da-8361-4b6b-8b92-bbe46b371c70" editable=false slideshow={"slide_type": ""}
%%capture

coefs = np.array([4, 6, 0, 1])
const = 42.0
assert np.allclose(
    antiderivative(coefs, const),
    np.array([1., 2., 0., 1., 42.])
), 'Тест не пройден.'

coefs = np.array([1, 7, -12, 21, -6])
const = 42.0
assert np.allclose(
    antiderivative(coefs, const),
    np.array([ 0.2, 1.75, -4., 10.5, -6., 42.])
), 'Тест не пройден.'

print("Выполнено")
score += 1
```

```python editable=false slideshow={"slide_type": ""}
print('Итоговый балл:', score)
```
