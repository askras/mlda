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

# Сверточные нейронные сети

Продолжительность работы: - 4 часа.

Мягкий дедлайн (10 баллов): 30.03.2027

Жесткий дедлайн (5 баллов): 13.04.2027


## Цель работы:
Изучить основные архитектурные особенности ключевых CNN (LeNet, AlexNet, ResNet, MobileNet) через их реализацию на Python/Keras.


## Пример выполнения типового задания Реализация LeNet-like сети:


```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

TF_ENABLE_ONEDNN_OPTS=0

# Загрузка данных MNIST
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train[..., np.newaxis] / 255.0
X_test = X_test[..., np.newaxis] / 255.0

# Создание модели с функциональным API
def build_lenet(input_shape=(28, 28, 1)):
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(6, kernel_size=5, activation='relu')(inputs)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(16, kernel_size=5, activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(120, activation='relu')(x)
    x = layers.Dense(84, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    return keras.Model(inputs, outputs)

model = build_lenet()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
history = model.fit(X_train, y_train, 
                    epochs=5, 
                    validation_split=0.2,
                    batch_size=128)

# Оценка точности
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc:.4f}')
```

### Классификация изображений с помощью обученной сети

```python editable=true slideshow={"slide_type": ""}
def plot_predictions(images, true_labels, pred_probs, n=5):
    plt.figure(figsize=(15, 5))
    for i in range(n):
        # Отображаем изображение
        plt.subplot(2, n, i+1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f"Истинное значение: {true_labels[i]}\nПредсказанное значение: {np.argmax(pred_probs[i])}")
        plt.axis('off')

        # Отображаем вероятности классов
        plt.subplot(2, n, i+n+1)
        bars = plt.bar(range(10), pred_probs[i], color='skyblue')
        bars[true_labels[i]].set_color('green')
        if np.argmax(pred_probs[i]) != true_labels[i]:
            bars[np.argmax(pred_probs[i])].set_color('red')
        plt.xticks(range(10))
        plt.ylim(0, 1)
        plt.xlabel("Класс")
        plt.ylabel("Вероятность")

        print()

    plt.tight_layout()
    plt.show()
```

```python
# Выбираем случайные изображения из тестового набора
indices = np.random.choice(len(X_test), 5, replace=False)
test_samples = X_test[indices]
true_labels = y_test[indices]

# Делаем предсказания
pred_probs = model.predict(test_samples)

# Визуализируем результаты
plot_predictions(test_samples, true_labels, pred_probs)
```

```python
# Пример классификации пользовательского изображения
def predict_custom_image(image_path):
    # Загрузка и предобработка изображения
    img = keras.preprocessing.image.load_img(
        image_path, 
        color_mode='grayscale', 
        target_size=(28, 28)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = img_array[:, :, 0:1] / 255.0  # Нормализация и добавление размерности канала

    # Предсказание
    pred = model.predict(np.array([img_array]))
    predicted_class = np.argmax(pred)


    # Визуализация
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(img_array.squeeze(), cmap='gray')
    plt.title(f"Your image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    bars = plt.bar(range(10), pred[0], color='skyblue') # вероятность классов
    bars[predicted_class].set_color('blue') # выделяем предсказанный класс
    plt.xticks(range(10))
    plt.xlabel("Класс")
    plt.ylabel("Вероятность")
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.show()

    return predicted_class
```

```python
# Пример использования
# В папке ./img есть файлы `0.png`, `2.png`, `7.png` для тестов
# Создайте полный комлект цифр
predicted_digit = predict_custom_image('./img/7.png')
print(f"Предсказанное значение: {predicted_digit}")
```

```python
# А еще есть файл `abrakadabra.png` :)
number_image = "./img/abrakadabra.png" # путь к вашему изображению
predicted_digit = predict_custom_image(number_image)
print(f"Предсказанное значение: {predicted_digit}")
```

<!-- #region -->
## Задание 1: Реализация AlexNet-like архитектуры


1. Постройте модель AlexNet для датасета CIFAR-10:
   - Входной слой: 32x32x3
   - Последовательно примените:
     - Conv(96, 11x11, stride 4) -> MaxPool(3x3, stride 2)
     - Conv(256, 5x5, padding='same') -> MaxPool(3x3, stride 2)
     - Conv(384, 3x3, padding='same')
     - Conv(384, 3x3, padding='same')
     - Conv(256, 3x3, padding='same') -> MaxPool(3x3, stride 2)
   - Полносвязные слои: 4096 -> 4096 -> 10
   - Добавьте Dropout(0.5) после каждого полносвязного слоя
2. Обучите модель и достигните точности > 65% на тестовом наборе
<!-- #endregion -->

## Задание 2: Визуализация Saliency Maps

1. Загрузите предобученную модель VGG16
2. Выберите 5 случайных изображений из набора CIFAR-10
3. Реализуйте расчет Saliency Map через градиенты по входу:
   - Используйте tf.GradientTape()
   - Нормализуйте значения градиентов
4. Визуализируйте исходные изображения и их Saliency Maps

```python
# Здесь должен быть ваш код
```

# Задание 3: Реализация Residual блока

1. Реализуйте residual block с bottleneck (1x1 -> 3x3 -> 1x1)
2. Постройте модель ResNet-18 для CIFAR-10:
   - Начальный слой: Conv(64, 7x7, stride 2) -> MaxPool(3x3)
   - 4 группы residual блоков: [2, 2, 2, 2] блоков в группе
   - Global Average Pooling -> Dense(10)
3. Сравните точность с обычной CNN

```python
# Здесь должен быть ваш код
```

# Задание 4: Сравнение MobileNet и CNN

1. Реализуйте две модели:
  - MobileNet-like архитектура с Depthwise Separable Conv
  - Обычная CNN с аналогичной глубиной
2. Обучите обе модели на Fashion-MNIST
3. Сравните:
   - Точность на тестовом наборе
   - Количество параметров
   - Время предсказания для 1000 изображений

```python
# Здесь должен быть ваш код
```

# Задание 5: Применение Grad-CAM

1. Реализуйте метод Grad-CAM для произвольной CNN
2. Для 3 классов ImageNet:
   - Загрузите изображения из интернета
   - Сгенерируйте Grad-CAM карты
   - Визуализируйте наложения карт на изображения
3. Объясните, какие части изображения важны для классификации

```python
# Здесь должен быть ваш код
```

## Рекомендации:
- Все модели создавайте через `keras.Input` и явное соединение слоев
- Используйте Depthwise Conv, Residual Connections, Bottleneck
- Для ускорения обучения используйте GPU (Если возможно)
- Для задач классификации применяйте аугментацию данных
- Экспериментируйте с разными оптимизаторами и learning rate


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

