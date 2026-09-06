---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.7
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Свёртки для обработки изображений

Продолжительность работы: - 4 часа.

Мягкий дедлайн (10 баллов): 27.03.2027

Жесткий дедлайн (5 баллов): 10.04.2027


## Цель работы:
Изучить основные операции свёртки, применяемые для обработки изображений, и научиться реализовывать их с использованием Python и библиотеки Keras.


## Пример выполнения типового задания:

**Задача**: Реализуйте операцию свёртки для сглаживания чёрно-белого изображения.

**Решение**:

```python
import numpy as np
from scipy.ndimage import convolve
from sklearn.datasets import load_sample_image
import matplotlib.pyplot as plt

# Загрузка изображения
image = load_sample_image("china.jpg")
gray_image = np.mean(image, axis=2)  # Преобразование в градации серого

# Ядро свёртки для сглаживания
kernel = np.ones((3, 3)) / 9

# Применение свёртки
smoothed_image = convolve(gray_image, kernel)

# Визуализация результатов
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Исходное изображение")
plt.imshow(gray_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Сглаженное изображение")
plt.imshow(smoothed_image, cmap='gray')
plt.axis('off')
plt.show()
```

## Задание 1: Выделение границ на изображении с помощью фильтра Собеля

**Описание**: Реализуйте операцию выделения границ на чёрно-белом изображении с помощью фильтра Собеля. Постройте карты вертикальных и горизонтальных границ.

```python
# Загрузка изображения
image = load_sample_image("china.jpg")
gray_image = np.mean(image, axis=2)  # Преобразование в градации серого

# Ядра Собеля для вертикальных и горизонтальных границ
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# Применение свёрток
edges_x = # Здесь должен быть Ваш код
edges_y = # Здесь должен быть Ваш код

# Объединение результатов
edges = np.sqrt(edges_x**2 + edges_y**2)

# Визуализация результатов
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Вертикальные границы")
plt.imshow(edges_x, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Горизонтальные границы")
plt.imshow(edges_y, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Объединённые границы")
plt.imshow(edges, cmap='gray')
plt.axis('off')
plt.show()
```

## Задание 2: Создание поточечной свёртки

**Описание**: Реализуйте поточечную свёртку (Pointwise Convolution) для преобразования числа каналов изображения.

```python
from tensorflow.keras import layers, Model
import numpy as np

# Генерация случайного тензора размерности (1, 28, 28, 3) для имитации изображения
input_tensor = np.random.rand(1, 28, 28, 3).astype(np.float32)

# Создание модели с поточечной свёрткой
inputs = layers.Input(shape=(28, 28, 3))
pointwise_conv = # Здесь должен быть Ваш код
model = Model(inputs=inputs, outputs=pointwise_conv)

# Проверка формы выходного тензора
output_tensor = model.predict(input_tensor)
print(f"Форма выходного тензора: {output_tensor.shape}")
```

## Задание 3: Реализация групповой свёртки

**Описание**: Реализуйте групповую свёртку (Grouped Convolution) с двумя группами для входного изображения.

```python
from tensorflow.keras import layers, Model

# Генерация случайного тензора размерности (1, 28, 28, 6) для имитации изображения
input_tensor = np.random.rand(1, 28, 28, 6).astype(np.float32)

# Создание модели с групповой свёрткой
inputs = layers.Input(shape=(28, 28, 6))
grouped_conv = # Здесь должен быть Ваш код
model = Model(inputs=inputs, outputs=grouped_conv)

# Проверка формы выходного тензора
output_tensor = model.predict(input_tensor)
print(f"Форма выходного тензора: {output_tensor.shape}")
```

## Задание 4: Реализация поканальной свёртки

**Описание**: Реализуйте поканальную свёртку (Depthwise Convolution) для обработки изображения.

```python
from tensorflow.keras import layers, Model

# Генерация случайного тензора размерности (1, 28, 28, 3) для имитации изображения
input_tensor = np.random.rand(1, 28, 28, 3).astype(np.float32)

# Создание модели с поканальной свёрткой
inputs = layers.Input(shape=(28, 28, 3))
depthwise_conv = # Здесь должен быть Ваш код
model = Model(inputs=inputs, outputs=depthwise_conv)

# Проверка формы выходного тензора
output_tensor = model.predict(input_tensor)
print(f"Форма выходного тензора: {output_tensor.shape}")
```

## Задание 5: Комбинация поканальной и поточечной свёрток

**Описание**: Реализуйте комбинацию поканальной и поточечной свёрток (Depthwise Separable Convolution).

```python
from tensorflow.keras import layers, Model

# Генерация случайного тензора размерности (1, 28, 28, 3) для имитации изображения
input_tensor = np.random.rand(1, 28, 28, 3).astype(np.float32)

# Создание модели с комбинацией поканальной и поточечной свёрток
inputs = layers.Input(shape=(28, 28, 3))
x = layers.DepthwiseConv2D(
    # Здесь должен быть Ваш код
)(inputs)  # Поканальная свёртка
x = layers.Conv2D(
    # Здесь должен быть Ваш код
)(x)  # Поточечная свёртка
model = Model(inputs=inputs, outputs=x)

# Проверка формы выходного тензора
output_tensor = model.predict(input_tensor)
print(f"Форма выходного тензора: {output_tensor.shape}")
```

## Критерии оценки:
1. Правильность реализации операций свёртки.
2. Корректность использования функционального API Keras.
3. Чёткость и понятность комментариев к коду.
4. Соответствие заданным требованиям (например, форма выходного тензора).


## Дополнительные материалы:
- [Keras Functional API](https://keras.io/guides/functional_api/)
- [Scipy Convolution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.convolve.html)
- [MNIST Dataset](https://keras.io/api/datasets/mnist/)
- [Convolutional Neural Networks](https://web.pdx.edu/~jduh/courses/Archive/geog481w07/Students/Ludwig_ImageConvolution.pdf)
- [Sobel Edge Detection](https://www.projectrhea.org/rhea/index.php/An_Implementation_of_Sobel_Edge_Detection)
