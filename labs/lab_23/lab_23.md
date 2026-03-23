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

# Сегментация изображений

Продолжительность работы: - 4 часа.

Мягкий дедлайн (10 баллов): 6.04.2026

Жесткий дедлайн (5 баллов): 20.04.2026

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

## Цель работы:
Целью данной лабораторной работы является изучение методов сегментации изображений, их практическое применение и реализация на Python с использованием библиотек Keras, scikit-learn, matplotlib и pandas. В ходе работы студенты освоят основные подходы к сегментации, включая пороговую обработку, кластеризацию и использование свёрточных нейронных сетей.


## Задания:


### Задание 1: Пороговая сегментация
Используйте метод пороговой обработки для выделения объектов на изображении.

```python
import cv2
import matplotlib.pyplot as plt

# Загрузка изображения (выберите свое изображение)
image = cv2.imread('./img/example.jpg', cv2.IMREAD_GRAYSCALE)

# Применение пороговой обработки
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Визуализация результата
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Оригинальное изображение")
plt.imshow(image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Пороговая сегментация")
plt.imshow(binary_image, cmap='gray')
plt.show()
```

```python
# Здесь должен быть ваш код для поиска оптимального порогового значения
```

### Задание 2: Кластеризация для сегментации
Реализуйте сегментацию изображения с использованием алгоритма k-means.

```python
from sklearn.cluster import KMeans
import numpy as np

# Преобразование изображения в одномерный массив
image = cv2.imread('./img/example.jpg')
pixel_values = image.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# Применение k-means
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(pixel_values)

# Восстановление сегментированного изображения
segmented_image = labels.reshape(image.shape[:2])

# Визуализация результата
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Оригинальное изображение")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 2, 2)
plt.title("Сегментация с помощью k-means")
plt.imshow(segmented_image, cmap='viridis')
plt.show()
```

```python
# Здесь должен быть ваш код для выбора оптимального количества кластеров
```

### Задание 3: Сегментация с использованием U-Net
Разберите пример использования архитектуры U-Net для сегментации изображений.

```python
# Пример:

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate

TF_ENABLE_ONEDNN_OPTS=0

def unet_model(input_size=(128, 128, 3)):
    inputs = Input(input_size)
    
    # Сжатие
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Расширение
    up1 = UpSampling2D(size=(2, 2))(pool2)
    concat1 = concatenate([conv2, up1], axis=-1)
    conv3 = Conv2D(64, 3, activation='relu', padding='same')(concat1)
    
    up2 = UpSampling2D(size=(2, 2))(conv3)
    concat2 = concatenate([conv1, up2], axis=-1)
    conv4 = Conv2D(1, 1, activation='sigmoid')(concat2)
    
    model = Model(inputs=[inputs], outputs=[conv4])
    return model

# Создание модели
model = unet_model()

# Компиляция модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

```python
# Здесь должен быть ваш код для загрузки данных и обучения модели
```

### Задание 4: Оценка качества сегментации
Реализуйте функцию для расчета метрик качества сегментации (IoU, Dice coefficient).

```python
def calculate_iou(y_true, y_pred):
    # Здесь должен быть ваш код 
    iou_score = ...
    return iou_score

def calculate_dice(y_true, y_pred):
    # Здесь должен быть ваш код
    dice_score = ...
    return dice_score

# Пример использования
y_true = np.random.randint(0, 2, size=(128, 128))
y_pred = np.random.randint(0, 2, size=(128, 128))

iou = calculate_iou(y_true, y_pred)
dice = calculate_dice(y_true, y_pred)

print(f"IoU: {iou}, Dice Coefficient: {dice}")
```

```python
# Здесь должен быть ваш код для сравнения различных методов сегментации
```

### Задание 5: Анализ результатов
Сравните результаты сегментации, полученные различными методами (пороговая обработка, k-means, U-Net). Проанализируйте их достоинства и недостатки.

```python
# Здесь должен быть ваш код для анализа и визуализации результатов
```

## Контрольные вопросы:
1. Что такое сегментация изображений? Какие задачи она решает?
2. Перечислите основные методы сегментации изображений.
3. В чем заключается разница между пороговой сегментацией и кластеризацией?
4. Какие метрики используются для оценки качества сегментации?
5. Какие архитектуры нейронных сетей чаще всего применяются для задач сегментации?

