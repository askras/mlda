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

<!-- #region id="view-in-github" colab_type="text" editable=true slideshow={"slide_type": ""} -->
# Введение в нейронные сети

Продолжительность работы: - 4 часа.

Мягкий дедлайн (10 баллов): 02.03.2025

Жесткий дедлайн (5 баллов): 16.03.2025

<!-- #endregion -->

<!-- #region id="vb8ObdAxEYM5" editable=true slideshow={"slide_type": ""} -->
## Реализация логических функций с помощью искусственного нейрона
<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""}
class Neuron:
    """Искусственный нейрон"""

    def __init__(self, weights, bias):
        self.weights = weights     # Веса
        self.bias = bias           # Смещение

    def activate(self, inputs):
        """Передаточная функция нейрона"""
        # Вычисляем линейную комбинацию входов и весов, затем добавляем смещение
        total_input = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
        # Пороговая функция активации (Heaviside)
        return 1 if total_input >= 0 else 0
```

<!-- #region id="BlpzKCPfJzlc" editable=true slideshow={"slide_type": ""} -->
### Задание 1 (Выполнено). Используя представленную выше реализацию нейрона, реализуйте нейрон для вычисления логической функции конъюнкции (AND)
<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""}
# Настройка нейрона для функции AND
weights = [1, 1] # Веса для двух входов
bias = -1.5 # Смещение, необходимое для реализации функции AND

neuron_and = Neuron(weights, bias)

# Примеры входов
inputs = [
    [0, 0],  # 0 AND 0 = 0
    [0, 1],  # 0 AND 1 = 0
    [1, 0],  # 1 AND 0 = 0
    [1, 1]   # 1 AND 1 = 1
]

# Проверка выходов
for x in inputs:
    output = neuron_and.activate(x)
    print(f'Вход: {x}, Выход: {output}')
```

<!-- #region id="1IY-H_WXC4xp" editable=true slideshow={"slide_type": ""} -->
### Задание 2. Используя представленную выше реализацию нейрона, реализуйте нейрон для вычисления логической функции дизъюнкции (OR)
<!-- #endregion -->

```python id="1IY-H_WXC4xp" editable=true slideshow={"slide_type": ""}
# Настройка нейрона для функции OR
weights = [...] # Веса для двух входов
bias = ... # Смещение, необходимое для реализации функции AND

neuron_or = Neuron(weights, bias)

# Примеры входов
inputs = [
    [0, 0],  # 0 OR 0 = 0
    [0, 1],  # 0 OR 1 = 1
    [1, 0],  # 1 OR 0 = 1
    [1, 1]   # 1 OR 1 = 1
]

# Проверка выходов
for x in inputs:
    output = neuron_or.activate(x)
    print(f'Вход: {x}, Выход: {output}')
```

<!-- #region id="1IY-H_WXC4xp" editable=true slideshow={"slide_type": ""} -->
### Задание 3. Используя представленную выше реализацию нейрона, реализуйте нейрон для вычисления логической функции отрицания (NOT)
<!-- #endregion -->

```python id="1IY-H_WXC4xp" editable=true slideshow={"slide_type": ""}
# Настройка нейрона для функции NOT
weights = [...] # Веса для двух входов
bias = ... # Смещение, необходимое для реализации функции AND

neuron_not = Neuron(weights, bias)

# Примеры входов
inputs = [
    [0],  # NOT 0 = 1
    [1],  # NOT 1 = 0
]

# Проверка выходов
for x in inputs:
    output = neuron_not.activate(x)
    print(f'Вход: {x}, Выход: {output}')
```

<!-- #region id="1IY-H_WXC4xp" editable=true slideshow={"slide_type": ""} -->
### Задание 4. Используя представленную выше реализацию нейрона, реализуйте функцию исключающего ИЛИ (XOR)
<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""}
# Реализация XOR комбинацией двух нейронов
# x_1 XOR x_2 = (x_1 AND NOT x_2) OR (NOT x_1 AND x_2)

# Здесь должен быть ваш код
```

```python editable=true slideshow={"slide_type": ""}
# Реализация XOR используя один нейрон
# https://yandex.ru/patents/doc/RU2269155C2_20060127

# Здесь должен быть ваш код
```

<!-- #region id="GY8do4_iCe_1" editable=true slideshow={"slide_type": ""} -->
## Реализация обучения нейрона
<!-- #endregion -->

```python id="StJ2cxIHC2mn" editable=true slideshow={"slide_type": ""}
import random

def init_weights(num_weights=2):
    """Инициализирует и возвращает начальные веса для модели.

    Веса выбираются случайным образом из диапазона [0, 1).

    Параметры:
        num_weights (int): Количество весов, которое нужно сгенерировать. По умолчанию равно 2.

    Возвращает:
        Список случайных весов размером num_weights.
    """
    return [random.random() for _ in range(num_weights)]

def predict(x, w):
    """Вычисляет предсказание модели.

    Параметры:
        x (list): Входные данные (признаки).
        w (list): Веса модели.

    Возвращает:
        int: Результат классификации (1 или 0).
    """
    summator = x[0] * w[0] + x[1] * w[1]
    return 1 if summator >= 1 else 0

def train(x_train, y_train, w, speed):
    """Обучает модель, корректируя веса на основе ошибки предсказания.

    Параметры:
        x_train (list): Список входных данных для обучения.
        y_train (list): Список правильных ответов.
        w (list): Текущие веса модели.
        speed (float): Скорость обучения (коэффициент корректировки весов).
    """
    for i in range(len(x_train)):
        error = y_train[i] - predict(x_train[i], w)
        if error != 0:
            w[0] = w[0] + error * x_train[i][0] * speed
            w[1] = w[1] + error * x_train[i][1] * speed

def epoch(count, func_train, x_train, y_train, w, speed):
    """Выполняет несколько эпох обучения.

    Параметры:
        count (int): Количество эпох.
        func_train (function): Функция для обучения.
        x_train (list): Список входных данных для обучения.
        y_train (list): Список правильных ответов.
        w (list): Текущие веса модели.
        speed (float): Скорость обучения.
    """
    print("Параметры весов:\n")
    for i in range(count):
        func_train(x_train, y_train, w, speed)
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
### Задание 5 (Выполнено). Используя представленную выше модель обучения, подберите параметры нейрона для реализации логической функции конъюнкции (AND)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="_n1Pl5fNCfTF" outputId="2415a358-7b7b-429f-c1a1-39dacaed490c" editable=true slideshow={"slide_type": ""}
speed = 0.3  # Скорость обучения
w = init_weights()  # Инициализация весов
print(f"Стартовые веса: {w}")

# Данные для обучения
x_train = [[1, 1], [1, 0], [0, 1], [0, 0]]
y_train = [1, 0, 0, 0]

# Запуск обучения
epoch(5, train, x_train, y_train, w, speed)
print(f"Конечные веса: {w}")

# Вывод результатов после обучения
print("x_1 x_2 AND")
for x in x_train:
    print(f"{x[0]:3} {x[1]:3} {predict(x, w):3}")
```

<!-- #region colab={"base_uri": "https://localhost:8080/"} id="8d9L3zVLm1dD" outputId="73c85486-90ab-49f9-d81c-5160501680e1" editable=true slideshow={"slide_type": ""} -->
### Задание 6. Используя представленную выше модель обучения, подберите параметры нейрона для реализации логической функции дизъюнкции (OR)
<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""}
# Здесь должен быть ваш код
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
### Задание 7. Используя представленную выше модель обучения, подберите параметры нейрона для реализации логической функции отрицания (NOT)
<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""}
# Здесь должен быть ваш код
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
### Задание 8. Используя представленную выше модель обучения, подберите параметры нейрона для отрицания конъюнкции (NOT(x_1 AND x_2))
<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""}
# Здесь должен быть ваш код
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
### Задание 9. Используя представленную выше модель обучения, подберите параметры нейрона для отрицания дизъюнкции (NOT(x_1 OR x_2))
<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""}
# Здесь должен быть ваш код
```
