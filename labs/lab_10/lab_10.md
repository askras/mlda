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

<!-- #region id="view-in-github" colab_type="text" editable=true slideshow={"slide_type": ""} -->
## Ансамблевые методы в машинном обучении

Продолжительность работы: - 4 часа.

Мягкий дедлайн (10 баллов): 18.12.2025

Жесткий дедлайн (5 баллов): 25.12.2025
<!-- #endregion -->

<!-- #region id="Km92zcbQ359V" editable=true slideshow={"slide_type": ""} -->
**Задание 1: Классификация лиц из набора данных Olivetti Faces с помощью алгоритмов градиентного бустинга** 

В этом задании необходимо применитьм несколько популярных алгоритмов градиентного бустинга - LightGBM, XGBoost и GradientBoosting из библиотеки scikit-learn для решения задачи классификации лиц из набора данных [Olivetti Faces](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html). 

Цель - определить, какому человеку принадлежит новое изображение лица, на основе обучения на размеченных данных.
<!-- #endregion -->

<!-- #region id="-XR0Lzn73som" editable=true slideshow={"slide_type": ""} -->
Шаги выполнения задания:

1. Загрузите набор данных Olivetti Faces с помощью функции `fetch_olivetti_faces()` из scikit-learn.
2. Разделите данные на обучающую и тестовую выборки с помощью `train_test_split()`.
3. Выполните предобработку изображений:
   - Преобразуйте изображения в вектора признаков (flatten)
   - Нормализуйте значения пикселей (разделите на 255)
4. Создайте и обучите модели градиентного бустинга:
   - LGBMClassifier
   - XGBClassifier
   - GradientBoostingClassifier
5. Оцените качество классификации каждой модели на тестовой выборке, используя метрику accuracy, отчет классификации и построение матрицы ошибок.
6. Сравните точность и время обучения разных алгоритмов. Определите, какой из них лучше подходит для данной задачи.
7. Сделайте выводы о сравнительной эффективности разных алгоритмов градиентного бустинга для классификации лиц.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 220} id="iLRV7yOEvcIo" outputId="4a20ddbf-fdaa-45d9-e1c7-7b0ff238f4aa" editable=true slideshow={"slide_type": ""}
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces

# Загрузка датасета
faces = fetch_olivetti_faces()
images = faces.images  # Изображения размером 64x64
labels = faces.target  # Метки классов (номера людей)

# Отображение первых пяти изображений и их классов
fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for i in range(5):
    image = images[10*i]
    label = labels[10*i]
    axes[i].imshow(image, cmap='gray')
    axes[i].set_title(f'Класс {label}')
    axes[i].axis('off')

plt.tight_layout()
plt.show();
```

```python id="9FgnjFgUgs3m" editable=true slideshow={"slide_type": ""}
# Здесь должен быть ваш код
```

<!-- #region id="Pme9_0Ee2r49" editable=true slideshow={"slide_type": ""} -->
**Задание 2: Классификация лиц из набора Olivetti Faces с помощью ансамбля классификаторов [VotingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html)**

- В этом задании необходимо применить ансамблевый метод классификации VotingClassifier из библиотеки scikit-learn для решения задачи распознавания лиц из набора данных [Olivetti Faces](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html).
- VotingClassifier объединяет предсказания нескольких базовых классификаторов путем голосования, что позволяет улучшить качество классификации по сравнению с отдельными моделями
- Необходимо использовать несколько "слабых" классификаторов, таких как LogisticRegression, DecisionTreeClassifier и KNeighborsClassifier, NaiveBayes, объединив их в ансамбль с помощью VotingClassifier.
- Не забудьте предварительно обработать изображения перед подачей в модели:
  - Преобразовать изображения в вектора признаков (flatten)
  - Нормализовать значения пикселей (разделить на 255)
- После обучения, оцените качество классификации каждой модели на тестовой выборке, используя метрику accuracy, отчет классификации и построение матрицы ошибок.
- Сравните точность ансамбля с отдельными базовыми моделями и оцените эффект от их комбинирования.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 220} id="v7qv_oR7zxW3" outputId="cca0fc55-64b2-43bb-ab01-266ea5fb7236" editable=true slideshow={"slide_type": ""}
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces

# Загрузка датасета
faces = fetch_olivetti_faces()
images = faces.images  # Изображения размером 64x64
labels = faces.target  # Метки классов (номера людей)

# Отображение первых пяти изображений и их классов
fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for i in range(5):
    image = images[10*i]
    label = labels[10*i]
    axes[i].imshow(image, cmap='gray')
    axes[i].set_title(f'Класс {label}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()
```

```python id="5JTCgOGj033x" editable=true slideshow={"slide_type": ""}
# Здесь должен быть ваш код
```

<!-- #region id="NpoJ7k407pwz" editable=true slideshow={"slide_type": ""} -->
**Задание 3: Классификация лиц из набора Olivetti Faces с помощью ансамбля классификаторов** [**StackingClassifier**](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html)

- В этом задании необходимо применить ансамблевый метод классификации StackingClassifier из библиотеки scikit-learn для решения задачи распознавания лиц из набора данных [Olivetti Faces](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html).
-  StackingClassifier объединяет предсказания нескольких базовых классификаторов путем обучения метаклассификатора на их выходах, что позволяет улучшить качество классификации по сравнению с отдельными моделями и методом голосования VotingClassifier.
- Необходимо использовать несколько "слабых" классификаторов первого уровня, таких как LogisticRegression, DecisionTreeClassifier, KNeighborsClassifier и GaussianNB. Затем обучить метаклассификатор второго уровня (например, LogisticRegression или SVC) на выходах классификаторов первого уровня. Метаклассификатор будет учиться комбинировать предсказания базовых моделей оптимальным образом.
- Не забудьте предварительно обработать изображения перед подачей в модели:
  - Преобразовать изображения в вектора признаков (flatten)
  - Нормализовать значения пикселей (разделить на 255)
- После обучения, оцените качество классификации каждой модели на тестовой выборке, используя метрику accuracy, отчет классификации и построение матрицы ошибок.
- Сравните точность ансамбля с отдельными базовыми моделями и оцените эффект от их комбинирования.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 220} id="KKDImQ-H2vxr" outputId="7628eb70-f717-452f-d166-a816b5519330" editable=true slideshow={"slide_type": ""}
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces

# Загрузка датасета
faces = fetch_olivetti_faces()
images = faces.images  # Изображения размером 64x64
labels = faces.target  # Метки классов (номера людей)

# Отображение первых пяти изображений и их классов
fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for i in range(5):
    image = images[10*i]
    label = labels[10*i]
    axes[i].imshow(image, cmap='gray')
    axes[i].set_title(f'Класс {label}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()
```

```python id="RKlufzuJ2v26" editable=true slideshow={"slide_type": ""}
# Здесь должен быть ваш код
```

<!-- #region id="-lcS45XkxkDI" editable=true slideshow={"slide_type": ""} -->
**Задание 4 (необязательное): Предсказание нижней половины лица по верхней с помощью алгоритмов градиентного бустинга**

- В этом задании необходимо применить алгоритмы градиентного бустинга - LightGBM, XGBoost и GradientBoostingRegressor - для решения интересной задачи регрессии на датасете лиц [Olivetti Faces](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html).
- Цель - предсказать значения пикселей нижней половины лица по пикселям верхней половины.
- Каждое изображение лица представляет собой матрицу 64x64 в оттенках серого.
- Разделите изображения на верхнюю и нижнюю половины и обучите модельл предсказывать значения яркости пикселей нижней половины по значениям верхней.
- Сравните качество (по метрике $R^2$) и время обучения разных реализаций градиентного бустинга на этой задаче восстановления изображений.
- Посмотрите, насколько хорошо алгоритмы бустинга смогут достроить нижнюю часть лица по верхней.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 270} id="WW49Zf-RuE_J" outputId="7c785375-dc61-4b9d-9341-3efbe0f9b452" editable=true slideshow={"slide_type": ""}
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces

# Загрузка датасета
faces = fetch_olivetti_faces()
images = faces.images  # Изображения размером 64x64
n_samples = images.shape[0]

# Разделение изображений на верхнюю и нижнюю половины
upper_half = images[:, :32, :]  # Верхняя половина (первые 32 строки)
lower_half = images[:, 32:, :]  # Нижняя половина (последние 32 строки)

# Преобразование 2D изображений в 1D векторы
X = upper_half.reshape((n_samples, -1))
y = lower_half.reshape((n_samples, -1))

# Выбор одного примера для отображения
sample_index = 0  # Измените значение для выбора другого изображения

# Восстановление изображений из векторов
upper_face = X[sample_index].reshape(32, 64)
lower_face = y[sample_index].reshape(32, 64)

# Отображение верхней и нижней половин лица
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(upper_face, cmap='gray')
axes[0].set_title('Верхняя половина лица (X)')
axes[0].axis('off')

axes[1].imshow(lower_face, cmap='gray')
axes[1].set_title('Нижняя половина лица (y)')
axes[1].axis('off')

plt.show()
```

```python id="_Cj0wjm_hJM7" editable=true slideshow={"slide_type": ""}
# Здесь должен быть ваш код
```
