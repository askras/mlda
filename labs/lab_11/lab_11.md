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

<!-- #region id="0j_kTt7qVxeX" editable=true slideshow={"slide_type": ""} -->
## Random forest (Случайный лес)

Продолжительность работы: - 4 часа.

Мягкий дедлайн (10 баллов): 18.12.2025

Жесткий дедлайн (5 баллов): 25.12.2025

<!-- #endregion -->

<!-- #region id="hwRT_4l_KybQ" editable=true slideshow={"slide_type": ""} -->
# Задание 1. Обучите классификатор Random Forest для решения задачи бинарной классификации: для каждого человека научиться предсказывать, выживет ли он при крушении Титаника.

Ссылка на датасет: https://www.kaggle.com/c/titanic/data
<!-- #endregion -->

<!-- #region id="6-SQIKsyWzDK" editable=true slideshow={"slide_type": ""} -->
## Устанавливаем зависимости
<!-- #endregion -->

```python id="rT_eTIY5W9_h" editable=true slideshow={"slide_type": ""}
# Здесь должен быть ваш код
```

<!-- #region id="AedM9KndWzMB" editable=true slideshow={"slide_type": ""} -->
## Импортируем датасет
<!-- #endregion -->

```python id="CJ3mE4JpXGZE" editable=true slideshow={"slide_type": ""}
# Здесь должен быть ваш код
```

<!-- #region id="h8LokBddKybd" editable=true slideshow={"slide_type": ""} -->
## Производим разведовательный анализ данных

Получим представление о данных в датасете:
<!-- #endregion -->

```python id="yq0aqSApXhTf" editable=true slideshow={"slide_type": ""}
# Здесь должен быть ваш код
```

<!-- #region id="-4VJEzN8Kybi" editable=true slideshow={"slide_type": ""} -->
## Сформируем обучающую и тестовую выборки:

<!-- #endregion -->

```python id="v40NR9NWXhy-" editable=true slideshow={"slide_type": ""}
# Здесь должен быть ваш код
```

<!-- #region id="mY9E-4bBKybn" editable=true slideshow={"slide_type": ""} -->
## Feature Engineering

Поработаем с признаками, выделим важные, изменим форму их представления (при надобности)
<!-- #endregion -->

```python id="HTK4tZTAYsQJ" editable=true slideshow={"slide_type": ""}
# Здесь должен быть ваш код
```

<!-- #region id="9Hpb-KlSZVMy" editable=true slideshow={"slide_type": ""} -->
## Построение базовой модели:
<!-- #endregion -->

```python id="_6GkXyKGZiMy" editable=true slideshow={"slide_type": ""}
# Здесь должен быть ваш код
```

<!-- #region id="n-W1QwERZ8Gn" editable=true slideshow={"slide_type": ""} -->
## Оценка точности модели:
<!-- #endregion -->

<!-- #region id="uDy7Y4E4am69" editable=true slideshow={"slide_type": ""} -->
### Напишите функцию, принимающую на вход аргументы y_pred, y_test и выполняющую визуализацию матрицы ошибок и отчета классификации
<!-- #endregion -->

```python id="O9h6cWIrZ9Le" editable=true slideshow={"slide_type": ""}
# Здесь должен быть ваш код
```

<!-- #region id="i41JeZ4jatU3" editable=true slideshow={"slide_type": ""} -->
### Оцените точность модели:
<!-- #endregion -->

```python id="J6mnzmr-a55M" editable=true slideshow={"slide_type": ""}
# Здесь должен быть ваш код
```

<!-- #region id="mGpNzrqKa-o-" editable=true slideshow={"slide_type": ""} -->
## Настройка гиперпараметров модели:
<!-- #endregion -->

<!-- #region id="ev_8vFTUbIQC" editable=true slideshow={"slide_type": ""} -->
RandomSearchCV
<!-- #endregion -->

```python id="-KrLGqfEbFj6" editable=true slideshow={"slide_type": ""}
# Здесь должен быть ваш код
```

<!-- #region id="yfHfNr8mbJGg" -->
GridSearchCV
<!-- #endregion -->

```python id="LelJW7VdbKCu" editable=true slideshow={"slide_type": ""}
# Здесь должен быть ваш код
```

<!-- #region id="e8ilf308rPB8" editable=true slideshow={"slide_type": ""} -->
## Оценка лучшей модели:
<!-- #endregion -->

```python id="Bg61rmBGrU94" editable=true slideshow={"slide_type": ""}
# Здесь должен быть ваш код
```

<!-- #region id="6yaKkCFGrbcK" editable=true slideshow={"slide_type": ""} -->
# Задание 2. Решите задачу из предыдущего пункта используя другие, ранее изученные классификаторы. Сравните их точность предсказания с Random Forest
<!-- #endregion -->

```python id="L6NtNp3Pr7IT" editable=true slideshow={"slide_type": ""}
# Здесь должен быть ваш код
```

<!-- #region id="Ok4UVapNuBEr" editable=true slideshow={"slide_type": ""} -->
# Задание 2.1 (Не обязательно). Реализуйте функцию, для взаимодействия с обученной Вами моделью

* Функция должна принимать на вход обученную модель классификатора;

* После вызова функции, у пользователя через консоль запрашиваются значения признаков. При запросе значений нужно вывести пояснения о типе и диапазоне возможных значений. Также реализуйте обработку исключений;

* После ввода значений для всех признаков в консоль, выводится результат работы классификатора.
<!-- #endregion -->

```python id="R45z2E1KwT0M" editable=true slideshow={"slide_type": ""}
# Здесь должен быть ваш код

```
