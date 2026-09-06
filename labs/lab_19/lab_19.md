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

# Регуляризация нейронных сетей

Продолжительность работы: - 4 часа.

Мягкий дедлайн (10 баллов): 13.03.2027

Жесткий дедлайн (5 баллов): 27.03.2027


## Пример решения: L2 регуляризация на MNIST

- `Input()`: Создает входной тензор формы (784,)
- `Dense()`: Слой с L2 регуляризацией (штраф за большие веса)
- Явное соединение слоев через оператор вызова `()`

```python
# Импорт библиотек и загрузка данных
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras import Input, Model
from keras.layers import Dense
from keras import regularizers
```

```python
# Загрузка и предобработка данных
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 784).astype('float32') / 255.0
X_test = X_test.reshape(-1, 784).astype('float32') / 255.0
```

```python
# Определение архитектуры через Functional API

input_layer = Input(shape=(784,))
x = Dense(128, activation='relu', 
          kernel_regularizer=regularizers.l2(0.01))(input_layer)
output_layer = Dense(10, activation='softmax')(x)

# Сборка модели
model = Model(inputs=input_layer, outputs=output_layer)

# Визуализация архитектуры
model.summary()
```

```python
# Компиляция и обучение

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

```python
# Обучение с валидацией
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=256,
    validation_split=0.2,
    verbose=1
)
```

```python
# Визуализация результатов
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss Curves')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy Curves')
plt.legend()
plt.show()
```

```python
# Оценка модели
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f'Test accuracy: {test_acc:.4f}')
print(f'Test loss: {test_loss:.4f}')

```

## Задание 1: Сравнение регуляризаций

Реализовать L1 и L2 регуляризацию через

```python
from keras import Input, Model
from keras.layers import Dense

def build_model(regularizer):
    inputs = Input(shape=(784,))
    x = Dense(128, activation='relu', 
              kernel_regularizer=regularizer)(inputs)
    outputs = Dense(10, activation='softmax')(x)
    return Model(inputs, outputs)

# Здесь должен быть ваш код:
# 1. Создайте model_l1 с regularizers.l1(0.01)
# 2. Создайте model_l2 с regularizers.l2(0.01)
# 3. Обучите обе модели
# 4. Сравните их accuracy на тестовом наборе
```

## Задание 2: Ранняя остановка с кастомизацией

Реализуйте EarlyStopping с восстановлением весов

```python
from keras.callbacks import EarlyStopping

inputs = Input(shape=(784,))
x = Dense(256, activation='relu')(inputs)
outputs = Dense(10, activation='softmax')(x)
model = Model(inputs, outputs)

early_stop = EarlyStopping(
    monitor='# Здесь должен быть ваш код (val_accuracy)',
    patience=5,
    restore_best_weights=# Здесь должен быть ваш код (True)
)

model.compile(# Здесь задайте optimizer и loss)

history = model.fit(
    X_train, y_train,
    epochs=50,
    validation_split=0.2,
    callbacks=[early_stop]
)

# Здесь должен быть ваш код:
# Выведите номер эпохи остановки: len(history.history['loss'])
```

## Задание 3: Расширенная аугментация

Создайте комплексный генератор данных

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=25,
    # Здесь добавьте параметры:
    # zoom_range=,
    # width_shift_range=,
    # shear_range=,
)
```

```python
# Визуализация аугментаций
fig, axes = plt.subplots(2, 5, figsize=(15, 5))
for images, _ in datagen.flow(X_train.reshape(-1,28,28,1), y_train, batch_size=10):
    for i in range(5):
        axes[0,i].imshow(images[i].reshape(28,28))
    break
plt.show()
```

## Задание 4: Многоуровневый Dropout

Реализуйте модель с каскадным dropout

```python
inputs = Input(shape=(784,))
x = Dense(512, activation='relu')(inputs)
x = # Здесь добавьте Dropout(0.4)
x = Dense(256, activation='relu')(x)
x = # Здесь добавьте Dropout(0.3)
outputs = Dense(10, activation='softmax')(x)

model = Model(inputs, outputs)

model.compile(
    optimizer='adam',
    loss='# Здесь должен быть ваш код',
    metrics=['accuracy']
)

# Обучите модель и сравните с базовой версией без dropout
```

## Задание 5: Оптимизация модели через прореживание

Примените техники сжатия модели

```python
import tensorflow_model_optimization as tfmot

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
```

```python
# Конфигурация прореживания
pruning_params = {
    'pruning_schedule': 
    tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.30,
        final_sparsity=0.70,
        begin_step=0,
        end_step=2000
    )
}
```

```python
# Создание модели для прореживания
inputs = Input(shape=(784,))
x = Dense(128, activation='relu')(inputs)
outputs = Dense(10, activation='softmax')(x)
model = Model(inputs, outputs)

model = prune_low_magnitude(# Здесь передайте модель и pruning_params)

model.compile(# Настройте optimizer и loss)
```

```python
# Обучение с callback'ами прореживания
model.fit(# Добавьте параметры обучения)

# Вывод статистики
stripped_model = tfmot.sparsity.keras.strip_pruning(model)
print(f"Параметры до: {model.count_params()}")
print(f"Параметры после: {stripped_model.count_params()}")
```

## Задание 6.

1. Сравните графики обучения для L1 и L2 регуляризации
2. Объясните, почему EarlyStopping можно считать формой регуляризации
3. Проанализируйте, как dropout влияет на время сходимости модели
4. Рассчитайте степень сжатия модели после прореживания
5. Предложите стратегию комбинирования методов регуляризации




