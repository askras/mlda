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

<!-- #region id="view-in-github" colab_type="text" editable=true slideshow={"slide_type": "slide"} -->
# Лекция 14: Случайный лес

МГТУ им. Н.Э. Баумана

Красников Александр Сергеевич

https://github.com/askras/mlda/

2024-2025
<!-- #endregion -->

<!-- #region id="zERY-QEMnvjq" editable=true slideshow={"slide_type": "slide"} -->
Решающие деревья для задач классификации и регрессии редко используются в чистом виде. Однако они популярны в ансамблевых методах, таких как бэггинг и бустинг:

- **Бэггинг** (от английского *bootstrap aggregating*) — это метод ансамблевого обучения, при котором несколько моделей (обычно одного типа) обучаются параллельно на разных подвыборках исходных данных, полученных методом бутстрепа (случайной выборки с возвращением). Предсказания этих моделей затем объединяются (например, путем усреднения или голосования), что снижает обобщенную ошибку модели за счет уменьшения вариативности.

- **Бустинг** — это метод ансамблевого обучения, в котором модели обучаются последовательно, и каждая следующая модель стремится исправить ошибки предыдущих. На каждом шаге обучению уделяется больше внимания тем данным, на которых предыдущие модели ошибались. Итоговый прогноз получается путем взвешенного объединения предсказаний всех моделей. Бустинг позволяет повысить точность модели за счет последовательного усиления слабых моделей.
<!-- #endregion -->

<!-- #region id="CVvaVnpvKybV" editable=true slideshow={"slide_type": "slide"} -->
## Random Forest (Случайный лес)

- **Random Forest (Случайный лес)** — алгоритм контролируемого обучения для задач классификации и регрессии. Это гибкий и простой в использовании метод, который строит множество деревьев решений на подвыборках данных, получает предсказания от каждого дерева и объединяет их путем голосования или усреднения. Random Forest также оценивает важность признаков.

- Алгоритм Random Forest объединяет несколько деревьев решений, образуя "лес", откуда и происходит название. В классификаторе Random Forest увеличение количества деревьев повышает точность модели.
<!-- #endregion -->

<!-- #region id="55iG2iEqpSAZ" editable=true slideshow={"slide_type": "slide"} -->
## Принцип алгоритма Random Forest

Алгоритм Random Forest состоит из двух основных этапов.

**Этап 1: Построение случайного леса**

1. **Создание множества деревьев:**
   - Повторяем следующие шаги `n` раз для создания `n` деревьев в лесу:
     - **Случайная выборка признаков:**
       - Случайным образом выбираем подмножество из `k` признаков из общего числа `m` признаков, где `k < m`.
     - **Построение дерева решений:**
       - Используя выбранные `k` признаков, находим наилучшую точку разделения для текущего узла.
       - Разбиваем узел на дочерние узлы по этому разделению.
       - Повторяем процесс для каждого дочернего узла, пока не будут выполнены условия остановки (например, достигнута максимальная глубина или минимальное количество узлов `l`).


<!-- #endregion -->

<!-- #region id="N7vxr9wyKybW" editable=true slideshow={"slide_type": "slide"} -->
**Этап 2: Прогнозирование с помощью леса**

1. **Применение леса к новым данным:**
   - Для каждого нового примера данных пропускаем его через все `n` деревьев в лесу.
   - Получаем предсказание от каждого дерева.
2. **Объединение предсказаний:**
   - **Для классификации:**
     - Применяем голосование большинства и выбираем класс, который был предсказан большинством деревьев.
   - **Для регрессии:**
     - Вычисляем среднее значение предсказаний всех деревьев.

Таким образом, алгоритм Random Forest объединяет результаты множества деревьев решений, каждое из которых обучено на случайном подмножестве признаков, что позволяет улучшить точность и устойчивость модели.
<!-- #endregion -->

<!-- #region id="Cb9osU9TpTCS" editable=true slideshow={"slide_type": "slide"} -->
![](./img/random_forest.png)
<!-- #endregion -->

<!-- #region id="QuqbM4FOKybX" editable=true slideshow={"slide_type": "slide"} -->
## Достоинства алгоритма Random Forest

- **Высокая точность прогнозов**: часто работает лучше линейных методов и сопоставим с бустингом.
- **Устойчивость к выбросам**: благодаря случайному выбору выборок методом бутстрэп-сэмплирования.
- **Нечувствительность к масштабированию признаков**: хорошо работает без нормализации данных.
- **Простота использования**: не требует тщательной настройки параметров и дает хорошие результаты "из коробки".
- **Эффективность на больших данных**: справляется с большим количеством признаков и классов.
- **Стойкость к переобучению**: добавление большего числа деревьев обычно улучшает модель.
- **Работа с пропущенными данными**: сохраняет точность даже при наличии пробелов в данных.
- **Возможности для анализа данных**: помогает в кластеризации, визуализации и обнаружении выбросов.
- **Легкость масштабирования**: можно увеличивать количество и глубину деревьев для улучшения модели.
<!-- #endregion -->

<!-- #region id="QuqbM4FOKybX" editable=true slideshow={"slide_type": "slide"} -->
## Недостатки алгоритма Random Forest

- **Сложность интерпретации**: результаты труднее понять по сравнению с одним деревом решений.
- **Неэффективность на разреженных данных**: хуже работает с данными, где много нулевых значений (например, в обработке текстов).
- **Отсутствие экстраполяции**: не умеет предсказывать значения за пределами обучающих данных, в отличие от линейной регрессии.
- **Возможность переобучения**: особенно на данных с большим количеством шума.
- **Предвзятость к признакам с многими категориями**: склонен уделять больше внимания признакам с большим числом уровней, что может искажать результаты.
- **Большой размер модели**: требует больше памяти для хранения большого количества деревьев (O(N \cdot K), где K – число деревьев).
<!-- #endregion -->

<!-- #region id="2wd55BFoKybY" editable=true slideshow={"slide_type": "slide"} -->
## Выбор признаков с помощью Random Forest

**Random Forest** не только используется для предсказаний, но и помогает определить, какие признаки наиболее влияют на результат модели. Это делается путем оценки важности каждого признака.

**Основная идея**: признаки, которые чаще используются для разделения узлов в деревьях и значительно уменьшают неопределенность (например, энтропию), считаются более важными.

**Процесс выбора признаков с помощью Random Forest:**

1. **Построение Random Forest**: создаётся множество решающих деревьев на обучающей выборке.

2. **Оценка важности признаков**: для каждого признака вычисляется его важноость на основе того, насколько сильно он уменьшает неопределенность при разделении узлов во всех деревьях леса.

3. **Ранжирование признаков**: признаки сортируются по убыванию их важности.

4. **Отбор признаков**: выбираются наиболее значимые признаки для дальнейшего использования в модели.

**Преимущества использования Random Forest для выбора признаков:**

- **Учет нелинейных взаимоотношений** между признаками.
- **Устойчивость к мультиколлинеарности** (когда признаки коррелируют между собой).
- **Интерпретируемость результатов** благодаря оценке важности каждого признака.
- **Сокращение размерности данных**, что может улучшить работу модели и снизить время обучения.
<!-- #endregion -->

<!-- #region id="T1beCuq0Kqnb" editable=true slideshow={"slide_type": "slide"} -->
## Пример 

Рассмотрим пример использования Random Forest для оценки важности признаков на наборе данных Iris (ирисы Фишера).
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 722} id="5aj485_OKjn-" outputId="b85aa8ab-313e-48bd-e0e5-3199e52c051a" editable=true slideshow={"slide_type": "slide"}
# Подключаем необходимые библиотеки для работы с данными и визуализации
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns

# Шаг 1: Загрузка данных
# Используем встроенный набор данных Iris из sklearn.datasets
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# Преобраование в DataFrame
#
data = pd.DataFrame(X, columns=feature_names)
data['species'] = y

# Шаг 2: Обучение модели Random Forest
# Cоздаём и обучаем модель RandomForestClassifier с 100 деревьями
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(data[feature_names], data['species'])

# Шаг 3: Оценка важности признаков
# Значения "важностей" признаков содержатся в атрибуте `feature_importances_`.
importances = rf.feature_importances_
feat_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Сортировка признаков по важности
feat_importances.sort_values(by='Importance', ascending=False, inplace=True)
display(feat_importances)

# Шаг 4: Визуализация важности признаков
# Cтроим график, отображающий важность каждого признака
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_importances)
plt.title('Важность признаков')
plt.xlabel('Значение важности')
plt.ylabel('Признак')
plt.show()
```

<!-- #region id="TxiCvOeVKwEA" editable=true slideshow={"slide_type": "slide"} -->
## Вывод

Используя Random Forest для оценки важности признаков, мы можем:

- **Идентифицировать наиболее значимые признаки**, влияющие на результат модели.
- **Сократить количество признаков**, удалив менее важные, что может упростить модель и снизить риск переобучения.
- **Улучшить интерпретируемость модели**, сосредоточившись на ключевых факторах.
<!-- #endregion -->

<!-- #region id="pt1vZIAZKybY" editable=true slideshow={"slide_type": "slide"} -->
## Разница между случайными лесами и деревьями решений

- **Построение модели**: Дерево решений — это одиночная модель, которая разделяет данные на основе признаков для предсказания. Случайный лес — ансамбль деревьев решений, где каждое дерево обучается на случайном подмножестве данных и признаков.

- **Ансамбль**: В случайном лесе множество деревьев голосуют за итоговое предсказание. Дерево решений предсказывает результат самостоятельно.

- **Стабильность и переобучение**: Деревья решений могут переобучаться, точно подстраиваясь под обучающие данные. Случайный лес более устойчив к переобучению благодаря усреднению результатов разных деревьев.

- **Скорость работы**: Деревья решений обучаются быстрее, так как создаётся одна модель. Случайный лес обучается дольше из-за множества деревьев, но предсказания могут выполняться быстрее за счёт параллельной обработки.

**Итог**: Случайный лес обеспечивает более высокую точность и устойчивость к переобучению, но требует больше времени на обучение. Деревья решений проще и быстрее в обучении, но могут быть менее точными и склонными к переобучению.
<!-- #endregion -->

<!-- #region id="D1OTe64GKybZ" editable=true slideshow={"slide_type": "slide"} -->
## Практический пример

Датасет **car_evaluation.csv** представляет собой популярный набор данных для задач классификации в машинном обучении. Он содержит информацию об автомобилях с различными харатеристиками, и его цель — оценить общую приемлемость автомобиля на основе этих харатеристик.
<!-- #endregion -->

```python id="iPZQ3WrpKyba" editable=true slideshow={"slide_type": "slide"}
import category_encoders as ce
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
```

```python id="povFCNKHKybd" editable=true slideshow={"slide_type": "slide"}
data = './car_evaluation.csv'

df = pd.read_csv(data)
df.sample(10)
```

<!-- #region id="UBpJV6qPM8vR" editable=true slideshow={"slide_type": "slide"} -->
**Признаки датасета:**

1. **buying**: цена покупки автомобиля.
   - Возможные значения: `vhigh` (очень высокая), `high` (высокая), `med` (средняя), `low` (низкая).

2. **maint**: Стоимость обслуживания автомобиля.
   - Возможные значения: `vhigh`, `high`, `med`, `low`.

3. **doors**: Количество дверей в автомобиле.
   - Возможные значения: `2`, `3`, `4`, `5more` (5 и более).

4. **persons**: Вместимость по количеству пассажиров.
   - Возможные значения: `2`, `4`, `more` (больше).

5. **lug_boot**: Размер багажника.
   - Возможные значения: `small` (маленький), `med` (средний), `big` (большой).

6. **safety**: Уровень безопасности автомобиля.
   - Возможные значения: `low` (низкий), `med` (средний), `high` (высокий).

7. **class**: Общая приемлемость автомобиля (целевой признак).
   - Возможные значения: `unacc` (неприемлемый), `acc` (приемлемый), `good` (хороший), `vgood` (очень хороший).
<!-- #endregion -->

<!-- #region id="h8LokBddKybd" editable=true slideshow={"slide_type": "slide"} -->
## Разведовательный анализ данных

Получим представление о данных в датасете:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="E1rXq4Z2Kybd" outputId="2b8a879b-64e5-4af6-a620-030f178e2edf" editable=true slideshow={"slide_type": "fragment"}
df.shape
```

<!-- #region id="0M7_mxtpKybf" editable=true slideshow={"slide_type": "slide"} -->
### Просмотр сводной информации о наборе данных
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="ipGqwHloKybf" outputId="b57c0478-9b41-4882-916f-e4471f12ea69" editable=true slideshow={"slide_type": "fragment"}
df.info()
```

<!-- #region id="KoIeevD6Kybf" editable=true slideshow={"slide_type": "slide"} -->
### Частотное распределение значений признаков
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 806} id="3ct8IitM8lFT" outputId="0e455c40-f90f-4363-ee6e-028b140e9795" editable=true slideshow={"slide_type": "slide"}
col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

fig, axs = plt.subplots(2, 4, figsize=(12, 8))

for i, col in enumerate(col_names):
    ax = axs[i // 4, i % 4]
    df[col].value_counts().plot(kind='bar', ax=ax)
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')

plt.tight_layout()
plt.show()
```

<!-- #region id="Tmu3NBRdKybh" editable=true slideshow={"slide_type": "slide"} -->
### Рассмотрим значения целевого столбца `class`:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 241} id="uoEdzoKGKybh" outputId="cc29fe5d-6a2e-4b13-be60-3751e84e4834" editable=true slideshow={"slide_type": "fragment"}
df['class'].value_counts()
```

<!-- #region id="YRWlXa_JKybh" editable=true slideshow={"slide_type": "slide"} -->
### Проверим датафрейм на содержание пустых ячеек
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 304} id="Lxb_qfLNKybi" outputId="d8eb9a4c-96a7-4e91-bf1e-4c42bf96e733" editable=true slideshow={"slide_type": "fragment"}
df.isnull().sum()
```

<!-- #region id="2vov2QYeKybi" editable=true slideshow={"slide_type": "fragment"} -->
Видно, что в наборе данных нет пропущенных значений.
<!-- #endregion -->

<!-- #region id="-4VJEzN8Kybi" editable=true slideshow={"slide_type": "slide"} -->
## Сформируем общую обучающую выборку:
<!-- #endregion -->

```python id="MJqxoEs8Kybi" editable=true slideshow={"slide_type": "fragment"}
X = df.drop(['class'], axis=1)

y = df['class']
```

<!-- #region id="jIgxX3VwKybj" editable=true slideshow={"slide_type": "slide"} -->
## Разделим общую выборку данных на обучающую и тестовую
<!-- #endregion -->

```python id="8BbvP2OtKybn" editable=true slideshow={"slide_type": "fragment"}
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

X_train.shape, X_test.shape
```

<!-- #region id="mY9E-4bBKybn" editable=true slideshow={"slide_type": "slide"} -->
## Feature Engineering

**Feature Engineering (Инженерия признаков)**  - это процесс преобразования исходных данных в полезные признаки, которые помогают нам лучше понять нашу модель и повысить ее предсказательную силу. Я проведу инженерию признаков для различных типов переменных.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 272} id="r0snrkXdKybo" outputId="3efb9c10-7403-4c11-d22a-58654020caf2" editable=true slideshow={"slide_type": "fragment"}
X_train.dtypes
```

<!-- #region id="JVxRYuPnKybo" editable=true slideshow={"slide_type": "slide"} -->
### Закодируем значения категориальных признаков в числовой формат
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="3IAtty5bKybo" outputId="21a9ca7b-d2b8-40b6-e063-9e9812439164" editable=true slideshow={"slide_type": ""}
X_train.head()
```

```python id="jtHvWE6DKybp" editable=true slideshow={"slide_type": "slide"}
encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="iiPIOkXKKybq" outputId="db5b422f-10ad-4dd9-b8a0-6d7f1bb6e083" editable=true slideshow={"slide_type": "slide"}
X_train.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="NM79Hlc9Kybq" outputId="f45bd801-0bd4-4ee0-cc15-df296aadc9b8" editable=true slideshow={"slide_type": "slide"}
X_test.head()
```

<!-- #region id="B5WzLQa-mqB-" editable=true slideshow={"slide_type": "slide"} -->
## Модель классификатора Random Forest

**RandomForestClassifier** - это классификатор, реализующий метод случайного леса. Он создает ансамбль решающих деревьев на основе случайной выборки признаков и случайной выборки объектов. Классификатор прогнозирует класс объекта, основываясь на голосовании решений деревьев.

Значения параметров классификатора RandomForestClassifier:

1. n_estimators - количество деревьев в случайном лесу (по умолчанию 100).
2. criterion - функция для измерения качества разбиения (по умолчанию "gini"). Возможные значения: "gini" и "entropy".
3. max_depth - максимальная глубина деревьев (по умолчанию None). Если None, то узлы будут расширяться до тех пор, пока все листы не станут однородными.
4. min_samples_split - минимальное количество образцов, необходимое для разделения внутреннего узла (по умолчанию 2).
5. min_samples_leaf - минимальное количество образцов, необходимое для быть листом (по умолчанию 1).
6. max_features - количество признаков, которые будут использоваться при делении (по умолчанию "auto"). Возможные значения: "auto", "sqrt", "log2" или целое число.
7. bootstrap - флаг, указывающий, должны ли образцы использоваться с повторениями при построении деревьев (по умолчанию True).
8. random_state - начальное состояние генератора случайных чисел (по умолчанию None).
9. n_jobs - количество параллельных задач для выполнения (по умолчанию None).

Для использования RandomForestClassifier необходимо создать экземпляр классификатора, указать необходимые параметры (если необходимо), а затем обучить модель на тренировочных данных с использованием метода fit(). После обучения, можно использовать методы predict() и predict_proba() для прогнозирования метки класса и вероятности соответствия классам соответственно.
<!-- #endregion -->

```python id="0nMVirh2Kybr" editable=true slideshow={"slide_type": "slide"}
# Инициализируем и обучаем классификатор Random Forest при использовании 10 деревьев решений
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=10, random_state=0)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
```

```python colab={"base_uri": "https://localhost:8080/"} id="6QJTV693lKGI" outputId="bf1968d5-43b8-444e-96d3-9027f0a6fc85" editable=true slideshow={"slide_type": "slide"}
# Произведем оценку точности
from sklearn.metrics import accuracy_score

print(f'Оценка точности модели при использовании 10 деревьев решений : {accuracy_score(y_test, y_pred):0.4f}')
```

<!-- #region id="63km0-Y0Kybs" editable=true slideshow={"slide_type": "slide"} -->
## Модель классификатора Random Forest с параметром n_estimators=100
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="pc6LcrrtKybs" outputId="91878277-3bc1-4c39-ab9a-7e9c1e06f9a6" editable=true slideshow={"slide_type": "slide"}
# увеличим количество деревьев решений и посмотрим, как это повлияет на точность

rfc_100 = RandomForestClassifier(n_estimators=100, random_state=0)
rfc_100.fit(X_train, y_train)
y_pred_100 = rfc_100.predict(X_test)

print(f'Оценка точности модели при использовании 100 деревьев решений : {accuracy_score(y_test, y_pred_100):0.4f}')
```

<!-- #region id="JtaijSN-Kybt" editable=true slideshow={"slide_type": "slide"} -->
Точность модели с 10 деревьями решений составляет 0.9247, а со 100 деревьями решений - 0,9457. Таким образом, как и ожидалось, точность увеличивается с ростом числа деревьев решений в модели.
<!-- #endregion -->

<!-- #region id="3IDvmZ2jKybt" editable=true slideshow={"slide_type": "slide"} -->
## Поиск важных признаков с помощью модели Random Forest

До сих пор мы использовали все признаки, заданные в модели. Теперь оставим только важные и построим модель с использованием этих признаков, обратим внимание на то, как это повлияет на точность.

Атрибут "feature_importances_" содержит информацию о важности признаков для обученной модели классификации или регрессии.

В случае классификации, значение "feature_importances_" показывает, насколько каждый признак внес вклад в прогнозы классификации. Чем выше значение, тем важнее признак.

В случае регрессии, значение "feature_importances_" показывает, насколько каждый признак внес вклад в предсказания регрессии. Чем выше значение, тем важнее признак.

Обычно, значения "feature_importances_" представлены в виде массива или списка, где каждый элемент соответствует важности определенного признака.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="ANCo_iHMofmg" outputId="635ff0e8-2fba-4fd8-f595-2d94ba4b79f1" editable=true slideshow={"slide_type": "slide"}
# Создадим модель Random Forest 
clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)

# Сформируем читабельную табличку с рейтингом важности каждого признака, где наиболее важные признаки имеют большую оценку, а наименее важные - меньшую
feature_scores = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
feature_scores
```

```python colab={"base_uri": "https://localhost:8080/", "height": 472} id="hhdDYuB-Kybu" outputId="0fdfb033-13d0-4d93-d078-dcdd771bc90e" editable=true slideshow={"slide_type": ""}
# Визуализируем оценки признаков с помощью matplotlib и seaborn.

sns.barplot(x=feature_scores, y=feature_scores.index)

plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")

plt.show()
```

<!-- #region id="IYSWxosnKybv" editable=true slideshow={"slide_type": "slide"} -->
## Обучение классификатора Random Forest на выбранных признаках

Теперь исключим наименее важный признак `doors`, переобучим модель и проверим его влияние на точность.
<!-- #endregion -->

```python id="cnaVxJFAKybv" editable=true slideshow={"slide_type": "slide"}
X = df.drop(['class', 'doors'], axis=1)

y = df['class']
```

```python id="pCKycO5EKybv" editable=true slideshow={"slide_type": "slide"}
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
```

```python id="M_GVPtGpKybw" editable=true slideshow={"slide_type": "slide"}
encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'persons', 'lug_boot', 'safety'])

X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)
```

```python colab={"base_uri": "https://localhost:8080/"} id="Q8GW4iCEKybw" outputId="81727ed3-a2c4-4a76-a19c-bb4f6f377e93" editable=true slideshow={"slide_type": "slide"}
clf = RandomForestClassifier(n_estimators=10, random_state=0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print('Оценка точности модели при удалении признака "doors" : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
```

<!-- #region id="mR0XNQt2Kybx" editable=true slideshow={"slide_type": "slide"} -->
* Точность модели после исключения признака `doors` составляет: 0,9264. Точность модели с учетом всех признаков была: 0,9247. Таким образом, мы видим, что точность модели повысилась.

* Вторым наименее важным признаком является `lug_boot`. Если исключить его и переобучить модель, то точность окажется равной 0,8546. Это значительное снижение точности. Поэтому её удалять мы не будем.
<!-- #endregion -->

<!-- #region id="BlJSACXeKybx" editable=true slideshow={"slide_type": "slide"} -->
## Метрики качества (напоминание)

**Матрица ошибок** — это инструмент для оценки качества моделей классификации. Она сравнивает реальные значения с предсказанными моделью и показывает, где модель совершает ошибки. Матрица выглядит так:

```
                          | Предсказано Положительное | Предсказано Отрицательное
---------------------------------------------------------------------------------
Фактически Положительное  |        True Positive (TP) |       False Negative (FN)
Фактически Отрицательное  |       False Positive (FP) |        True Negative (TN)
```

**Компоненты матрицы:**

1. **True Positive (TP)**: модель верно предсказала положительный класс.
2. **True Negative (TN)**: модель верно предсказала отрицательный класс.
3. **False Positive (FP)**: модель ошибочно предсказала положительный класс для отрицательного примера (ложное сраатывание).
4. **False Negative (FN)**: модель ошибочно предсказала отрицательный класс для положительного примера (пропуск).

Матрица ошибок позволяет вычислить важные метрики:

- **Точность (Precision)**: доля верно предсказанных положительных классов из всех предсказанных положительных.

  $ \text{Precision} = \frac{TP}{TP + FP} $

- **Полнота (Recall)**: доля верно предсказанных положительных классов из всех фактических положительных.

  $ \text{Recall} = \frac{TP}{TP + FN} $

- **F1-мера**: гармоническое среднее между точностью и полнотой.

  $ F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} $

Матрица ошибок помогает понять, насколько эффективно модель различает классы, и выявить области для улучшения модели.ть области для улучшения модели.слять другие метрики, такие как точность (precision), полнота (recall) и F1-мера. Они часто используются в задачах классификации для оценки производительности модели и подстройки их параметров.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 564} id="E4a0NFMTgUYo" outputId="880cab6c-1e40-4084-f917-a9ac59c81bfc" editable=true slideshow={"slide_type": "slide"}
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt=".0f", cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

<!-- #region id="A85YFu60Kyby" editable=true slideshow={"slide_type": ""} -->
## Отчет о классификации

**Отчет о классификации** предоставляет ключевые метрики качества модели для каждого класса: **точность** (precision), **полноту** (recall), **F1-меру** (f1-score) и **support** (количество образцов каждого класса в истинных значениях). Он также включает усредненные значения этих метрик.
<!-- #endregion -->

```python id="A85YFu60Kyby" editable=true slideshow={"slide_type": ""}
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```

<!-- #region id="A85YFu60Kyby" editable=true slideshow={"slide_type": ""} -->
**Метрики:**

- **Precision (точность)**: доля верно предсказанных положительных результатов из всех предсказанных положительных.

- **Recall (полнота)**: доля верно предсказанных положительных результатов из всех фактических положительных.

- **F1-score (F1-мера)**: гармоническое среднее между точностью и полнотой, учитывающее баланс между ними.

- **Support**: количество образцов каждого класса в истинных данных.

**Средние значения:**

- **Micro avg**: усреднение метрик по всем образцам, учитывает общий вклад каждого класса.

- **Macro avg**: простое среднее метрик по классам, не учитывает дисбаланс классов.

- **Weighted avg**: среднее метрик по классам с учетом количества образцов в каждом классе, учитывает дисбаланс.

**Отчет о классификации** помогает быстро оценить производительность модели по основным метрикам и понять, как она справляется с каждым классом.

<!-- #endregion -->

<!-- #region id="a63bb0e1" papermill={"duration": 0.021901, "end_time": "2023-02-01T13:12:05.284456", "exception": false, "start_time": "2023-02-01T13:12:05.262555", "status": "completed"} editable=true slideshow={"slide_type": ""} -->
## Подбор гиперпараметров Random Forest. Кросс-валидация

**Кросс-валидация** (Cross Validation) — это метод оценки качества модели машинного обучения, который помогает определить, насколько хорошо модель будет работать на новых, неиспользованных данных. Она заключается в разделении исходных данных на несколько частей (называемых фолдами), и последовательном обучении и тестировании модели на разных комбинациях этих частей.
<!-- #endregion -->

<!-- #region id="Jfm0vGAjPKMT" editable=true slideshow={"slide_type": "slide"} -->
**Простой пример:**

Представьте, что у вас есть набор данных из 100 образцов. Вы хотите проверить, насколько хорошо ваша модель будет предсказывать результаты на новых данных. Применим **k-fold кросс-валидацию** с **k = 5**:

1. **Разбиение данных:**
   - Разделите данные на 5 раных частей по 20 образцов.

2. **Итеративное обучение и тестирование:**
   - **Итерация 1:**
     - Обучите модель на частях 1-4 (80 образцов).
     - Протестируйте модель на части 5 (20 образцов).
   - **Итерация 2:**
     - Обучите модель на частях 1-3 и 5.
     - Протестируйте на части 4.
   - **Итерации 3-5:**
     - Повторите процесс, каждый раз оставляя одну из частей для тестирования, а остальные используйте для обучения.

3. **Оценка качества:**
   - После каждой итерации фиксируйте качество модели (например, точность).
   - В конце вычислите среднее значение качества по всем итерациям.

**Преимущества кросс-валидации:**

- **Надёжность оценки:** Позволяет получить более стабильную и надёжную оценку качества модели, учитывая вариативность данных.
- **Предотвращение переобучения:** Помогает выявить и избежать ситуаций, когда модель хорошо работает на обучающих данных, но плохо — на новых.
- **Оптимизация модели:** Используется для подбора гиперпараметров и выбора наилучшей модели среди нескольких.
<!-- #endregion -->

<!-- #region id="Gy9mI8RGyfjn" editable=true slideshow={"slide_type": "slide"} -->
### RandomizedSearchCV

**RandomizedSearchCV** — это метод случайного поиска гиперпараметров для моделей машинного обучения. В отличие от GridSearchCV, который перебирает все возможные комбинации, RandomizedSearchCV случайным образом выбирает комбинации гиперпараметров из заданных распределений. Это ускоряет процесс оптимизации и снижает риск переобучения, особенно при большом количестве гиперпараметров.

**Основные параметры RandomizedSearchCV:**

1. **estimator**: модель, которую нужно обучить.
2. **param_distributions**: словарь параметров с числовыми диапазонами, из которых случайно выбираются значения.
3. **n_iter**: количество итераций поиска (число случайных комбинаций параметров).
4. **scoring**: метрика для оценки качества моделей (строка с названием метрики или функция).
5. **cv**: число фолдов для кросс-валидации (по умолчанию 5).
6. **random_state**: начальное состояние генератора случайных чисел для воспроизводимости результатов.
7. **n_jobs**: количество параллельных задач (-1 использует все доступные ядра процессора).
8. **verbose**: уровень подробности вывода информации о процессе обучения.
9. **refit**: если `True`, модель переобучается на всех данных с лучшими найденными параметрами.
10. **return_train_score**: если `True`, возвращает оценки на обучающем наборе данных.
11. **error_score**: способ обработки ошибок при оценке модели (`'raise'` для выброса исключения или числовое значение).
12. **pre_dispatch**: количество задач для предварительной отправки в очередь выполнения.

**RandomizedSearchCV** возвращает модель с оптимальными гиперпараметрами, готовую для предсказания на новых данных.

**RandomizedSearchCV** — эффективный инструмент для оптимизации гиперпараметров, который позволяет быстро найти наилучшую модель машинного обучения. Он ускоряет процесс обучения и улучшает качество моделей, эффективно исследуя пространство параметров.
<!-- #endregion -->

```python id="a3473343" papermill={"duration": 861.878905, "end_time": "2023-02-01T13:26:27.235883", "exception": false, "start_time": "2023-02-01T13:12:05.356978", "status": "completed"} outputId="146554da-f075-40c4-f28d-07cb2ee49c33" editable=true slideshow={"slide_type": "slide"}
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Определяем список значений для параметра n_estimators (количество деревьев в лесу)
n_estimators = [int(x) for x in np.linspace(start=100, stop=500, num=10)]

# Определяем список значений для параметра max_depth (максимальная глубина дерева)
max_depth = [int(x) for x in np.linspace(10, 500, num=5)]
max_depth.append(None)  # Добавляем значение None для отсутствия ограничения глубины

# Определяем список значений для параметра max_leaf_nodes (максимальное количество листовых узлов)
max_leaf_nodes = [int(i) for i in range(1, 100)]

# Определяем пространство гиперпараметров для случайного поиска
random_grid = {
    'n_estimators': n_estimators,
    'max_depth': max_depth,
    'max_leaf_nodes': max_leaf_nodes,
}

# Создаем модель классификатора случайного леса с фиксированным random_state для воспроизводимости
rf_clf = RandomForestClassifier(random_state=0)

# Инициализируем RandomizedSearchCV для поиска оптимальных гиперпараметров
rf_cv = RandomizedSearchCV(
    estimator=rf_clf,                # модель для настройки
    param_distributions=random_grid, # словарь с параметрами для случайного поиска
    n_iter=300,                      # количество итераций (случайных наборов гиперпараметров)
    scoring='accuracy',              # метрика для оценки качества моделей
    cv=5,                            # количество фолдов для кросс-валидации
    verbose=1,                       # уровень детализации вывода
    random_state=42,                 # фиксированный сид для воспроизводимости результатов
    n_jobs=-1                        # использование всех доступных ядер процессора
)

# Обучаем модель RandomizedSearchCV на обучающих данных с перебором гиперпараметров
rf_cv.fit(X_train, y_train)

# Получаем оптимальные гиперпараметры после поиска
rf_best_params = rf_cv.best_params_
print(f"Лучшие параметры: {rf_best_params}")

# Создаем новый классификатор случайного леса с оптимальными гиперпараметрами
rf_clf = RandomForestClassifier(**rf_best_params)

# Обучаем модель с оптимальными гиперпараметрами на обучающих данных
rf_clf.fit(X_train, y_train)

# Предсказываем классы для тестовых данных
y_pred = rf_clf.predict(X_test)

# Выводим оценку точности модели на тестовых данных
print(f'Оценка точности модели при подборе параметров с помощью RandomizedSearchCV: {accuracy_score(y_test, y_pred):0.4f}')
```

```python id="N5GUcNEt0SGf" outputId="6d6e4117-0ce8-4880-e80b-28eb02cb4707" editable=true slideshow={"slide_type": "slide"}
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt=".0f", cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

```python id="9NnHyaJP0SPm" outputId="31d86c0c-1fea-48f3-f4bd-88db79cf7969" editable=true slideshow={"slide_type": "slide"}
print(classification_report(y_test, y_pred))
```

<!-- #region id="31ecf3cd" papermill={"duration": 0.021868, "end_time": "2023-02-01T13:26:27.281474", "exception": false, "start_time": "2023-02-01T13:26:27.259606", "status": "completed"} editable=true slideshow={"slide_type": "slide"} -->
`RandomizedSearchCV` позволил нам сузить диапазон для каждого гиперпараметра. Теперь, когда мы знаем, где сосредоточить поиск, мы можем явно указать все комбинации параметров, которые необходимо попробовать. Для этого используется метод `GridSearchCV`, который вместо случайной выборки из распределения оценивает все заданные нами комбинации.
<!-- #endregion -->

<!-- #region id="9adb7e01" papermill={"duration": 0.021513, "end_time": "2023-02-01T13:26:27.326069", "exception": false, "start_time": "2023-02-01T13:26:27.304556", "status": "completed"} editable=true slideshow={"slide_type": "slide"} -->
### GridSearchCV

**GridSearchCV** — инструмент из библиотеки scikit-learn для подбора оптимальных гиперпараметров модели через перебор **ВСЕХ** заданных комбинаций с использованием кросс-валидации. Он помогает найти наилучшие параметры для улучшения качества модели.

**Основные параметры GridSearchCV:**

1. **estimator**: модель, для которой настраиваются гиперпараметры (должна соответствовать интерфейсу оценщика scikit-learn).
2. **param_grid**: словарь или список словарей с параметрами и их возможными значениями для перебора.
3. **scoring**: метрика для оценки качества модели; если не указано, используется метод `score` по умолчанию.
4. **cv**: стратегия кросс-валидации; по умолчанию 5-кратная.
5. **refit**: если `True`, модель переобучается на всех данных с лучшими найденными параметрами (по умолчанию `True`).
6. **verbose**: уровень детализации вывода; чем больше значение, тем подробнее информация.
7. **n_jobs**: число параллельных задач; `-1` использует все доступные ядра процессора.
8. **return_train_score**: если `True`, возвращает оценки на обучающих данных; по умолчанию `False`.

**Как работает GridSearchCV:**

- Метод `fit` перебирает все комбинации параметров из `param_grid`, обучает модель на каждой из них и оценивает качество по метрике `scoring`.
- По завершении поиска возвращается модель с наилучшими параметрами.

**Результат:**

GridSearchCV предоставляет информацию о лучшей модели и ее оптимальных параметрах, найденных в процессе перебора.
<!-- #endregion -->

```python id="7f77bd10" papermill={"duration": 1829.392205, "end_time": "2023-02-01T13:56:56.740551", "exception": false, "start_time": "2023-02-01T13:26:27.348346", "status": "completed"} outputId="7fb7238e-4d27-442e-d0b9-21e345789cb6" editable=true slideshow={"slide_type": "slide"}
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Определяем список значений для параметра n_estimators (количество деревьев)
n_estimators = [int(i) for i in range(1, 201)]

# Определяем список значений для параметра max_depth (максимальная глубина дерева)
max_depth = [100, 150, 200, 250, 300, 350, 400, 450, 500]
max_depth.append(None)  # Добавляем значение None для отсутствия ограничения глубины

# Определяем список значений для параметра max_leaf_nodes (максимальное количество листьев)
max_leaf_nodes = [int(i) for i in range(1, 101)]

# Создаем словарь с параметрами для перебора в GridSearchCV
params_grid = {
    'n_estimators': n_estimators,
    'max_depth': max_depth,
    'max_leaf_nodes': max_leaf_nodes,
}

# Инициализируем классификатор случайного леса с фиксированным random_state для воспроизводимости
rf_clf = RandomForestClassifier(random_state=0)

# Настраиваем GridSearchCV для поиска наилучших гиперпараметров
rf_cv = GridSearchCV(
    estimator=rf_clf,        # модель для настройки
    param_grid=params_grid,  # словарь с параметрами для перебора
    scoring="accuracy",      # метрика для оценки качества моделей
    cv=5,                    # количество фолдов для кросс-валидации
    verbose=1,               # уровень детализации вывода
    n_jobs=-1                # использование всех доступных ядер процессора
)

# Обучаем модель GridSearchCV на обучающих данных
rf_cv.fit(X_train, y_train)

# Получаем лучшие найденные параметры
best_params = rf_cv.best_params_
print(f"Лучшие параметры: {best_params}")

# Обучаем окончательную модель с лучшими параметрами на обучающих данных
rf_clf = RandomForestClassifier(**best_params)
rf_clf.fit(X_train, y_train)

# Предсказываем классы для тестовых данных
y_pred = rf_clf.predict(X_test)

# Выводим оценку точности модели на тестовых данных
print(f'Оценка точности модели при подборе параметров с помощью GridSearchCV: {accuracy_score(y_test, y_pred):0.4f}')
```

```python id="vpfDoUFb1qx9" outputId="2469d0d8-f981-49af-b9a8-5a8563c7f6e3" editable=true slideshow={"slide_type": "slide"}
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt=".0f", cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

```python id="k37ti9gi1r1y" outputId="d2249c4c-760c-4267-d9d5-f086a3c49829" editable=true slideshow={"slide_type": "slide"}
print(classification_report(y_test, y_pred))
```

<!-- #region id="k8w7cPHg3vB4" editable=true slideshow={"slide_type": "slide"} -->
## Выводы

1. Оценка точности модели при использовании 10 деревьев решений (остальные параметры по-умолчанию): 0.9244

2. Оценка точности модели при использовании 100 деревьев решений (остальные параметры по-умолчанию): **0.9457**

3. Оценка точности модели при использовании 10 деревьев и удалении признака "doors" : 0.9264

4. Оценка точности модели при подборе параметров с помощью RandomizedSearchCV : 0.9299
- Лучшие параметры: {'n_estimators': 188, 'max_leaf_nodes': 76, 'max_depth': 500}


5. Оценка точности модели при подборе параметров с помощью GridSearchCV : **0.9370**
- Лучшие параметры: {'max_depth': 100, 'max_leaf_nodes': 19, 'n_estimators': 147}
<!-- #endregion -->


