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

<!-- #region id="african-marble" editable=true slideshow={"slide_type": ""} -->
# Логистическая регрессия и метод ближайших соседей
<!-- #endregion -->

**Цель работы:** получение практических навыков использования модели логистической регрессии и метода ближайших соседей.

Продолжительность работы: - 4 часа.

Мягкий дедлайн (10 баллов): 27.11.2025

Жесткий дедлайн (5 баллов): 04.12.2025

<!-- #region id="sought-brooklyn" -->
## Логистическая регрессия

Метод логистической регрессии является мощным инструментом для бинарной классификации, обеспечивая простоту в интерпретации и применении. Однако для достижения лучшей производительности требуется тщательный выбор признаков и проверка предположений модели.

- Логистическая регрессия — это статистический метод, используемый для бинарной классификации. Он предсказывает вероятность принадлежности объекта к определенному классу на основе входных признаков.
- Логистическая регрессия использует логистическую функцию (сигмоиду), чтобы преобразовать линейную комбинацию входных признаков в вероятность.
  - Модель логистической регрессии:
$$
\hat y = \sigma (Xw).
$$
  - Сигмоида меняется в пределах от 0 до 1 и имеет вид:
$$
\sigma(x) = \frac{1}{1+e^{-x}}.
$$
-- Функция потерь log-loss:
$$
L = -\frac{1}{\ell}\sum_{i = 1}^{\ell}(y_i\log(\hat y_i) + (1 - y_i)\log(1 - \hat y_i)),
$$
где $\ell$ - количество объектов.
-- Регуляризация вводится таким же образом, как это было в случае линейной регрессии. Например, функция потерь для $L$-$2$ регуляризации выглядит так:
$$
\bar{L}(X, w) = L(X, w) + \frac{1}{2}\lambda\|w\|^2_2.
$$
- Коэффициенты модели алгоритм находит с помощью метода максимального правдоподобия, который максимизирует вероятность наблюдаемых данных, исходя из модели.
- Классификация производится путем установки порога (например, 0.5) для вероятностей. Если предсказанная вероятность превышает порог, объект классифицируется в один класс, иначе — в другой.
- Логистическая регрессия используется в различных областях, например:
  - Медицина (для диагностики болезней);
  - Финансовые услуги (для оценки кредитного риска);
  - Маркетинг (для прогноза успеха рекламных кампаний).
- Преимущества:
  - Простота и интерпретируемость модели.
  - Быстрая и эффективная при малом количестве признаков.
  - Легко адаптируется для многоклассовой классификации с помощью методов, таких как "один против всех".
- Недостатки:
  - Предположение о линейности отношений в логарифмической шкале.
  - Неэффективна при наличии сложных нелинейных зависимостей без дополнительной обработки (например, полиномиальные признаки).
  - Чувствительность к мультиколлинеарности (высокой корреляции между независимыми переменными).

<!-- #endregion -->



<!-- #region id="separate-gateway" -->
## Метод ближайших соседей

Метод ближайших соседей (k-NN) является мощным инструментом для классификации и регрессии, однако его эффективность и точность зависят от правильного выбора параметров и методов обработки данных. Для многих задач может потребоваться предварительная обработка данных и оптимизация для достижения наилучших результатов.

- Для классификации нового объекта алгоритм k-NN ищет k ближайших к нему соседей в обучающей выборке, основываясь на заданной метрике расстояния.
- Класс нового объекта определяется по большинству голосов его ближайших соседей (например, класс, который чаще всего встречается среди соседей).
- Наиболее распространенными метриками расстояний являются:
  - Евклидово расстояние: 
$$d(p, q) = \sqrt{\sum_{i=1}^{n}(p_i - q_i)^2}$$    
  - Манхэттенское расстояние: 
$$d(p, q) = \sum_{i=1}^{n}|p_i - q_i|$$
  - Расстояние Минковского: 
$$d(p, q) = \left(\sum_{i=1}^{n}|p_i - q_i|^r\right)^{1/r}$$
где $r$ — параметр, определяющий тип расстояния).
- Параметр k (количество соседей) является критически важным для работы алгоритма:
  - Небольшие значения k могут приводить к переобучению (алгоритм очень чувствителен к шуму).
  - Большие значения k могут приводить к недообучению (общее усреднение данных).
- Новый объект классифицируется на основе классов его ближайших соседей. Каждому соседу присваивается вес по частоте его класса, и выбирается класс с наибольшим голосом.
- Взвешенные методы могут использоваться для улучшения точности классификации, где соседи, находящиеся ближе, могут иметь больший вес.
- Метод k-NN демонстрирует снижение эффективности с увеличением числа измерений, что называется "проклятием размерности". Это связано с тем, что расстояния между точками становятся менее различимыми в высоких дименсиях.
- Метод k-NN широко используется в:
  - Распознавании образов (например, в задачах компьютерного зрения);
  - Рекомендательных системах;
  - Медицинской диагностике;
  - Классификации текстов и многом другом.
- Преимущества:
  - Простота реализации и понимания.
  - Нет необходимости в обучении (алгоритм использует все доступные данные).
  - Гибкость в использовании разных метрик расстояний.
- Недостатки:
  - Высокая вычислительная сложность (особенно на больших объемах данных).
  - Чувствительность к выбору k и метрикам расстояния.
  - Проблема проклятия размерности.
<!-- #endregion -->

```python id="manual-launch"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
```

<!-- #region id="union-darwin" -->
## Практика
<!-- #endregion -->

<!-- #region id="abroad-hamburg" -->
Рассмотрим свойства логистической регрессии и метода опорных векторов на примере классического набора данных ["Ирисы Фишера"](https://ru.wikipedia.org/wiki/Ирисы_Фишера). Этот набор состоит из 150 наблюдений, каждое из которых представляет собой четыре измерения: длина наружной доли околоцветника (`sepal length`), ширина наружной доли околоцветника (`sepal width`), длина внутренней доли околоцветника (`petal length`), ширина внутренней доли околоцветника (`petal width`). Каждое наблюдение относится к одному из трёх классов ириса: `setosa`, `versicolor` или `virginica`. Задача состоит в том, чтобы по измерениям предсказать класс цветка.

<img src="./img/iris.png" alt="drawing" width="800"/>
<!-- #endregion -->

```python id="several-bradford"
from sklearn.datasets import load_iris
iris = load_iris(as_frame=True)

data = iris['data']
y = iris['target'].values
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"elapsed": 429, "status": "ok", "timestamp": 1694978156125, "user": {"displayName": "Sergey Korpachev", "userId": "09181340988160569540"}, "user_tz": -180} id="77068ebd-7569-4672-b6e6-fdfc55460fc5" outputId="b4727ad6-f798-4d20-f682-c0f742d39174" editable=true slideshow={"slide_type": ""}
data.head()
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 561, "status": "ok", "timestamp": 1694978158940, "user": {"displayName": "Sergey Korpachev", "userId": "09181340988160569540"}, "user_tz": -180} id="0dc8d8b2-e5fa-4bc4-9614-675d61e4f606" outputId="d10e35a8-506a-4b2c-e040-c3b060d1f9bb"
y[:5]
```

<!-- #region id="velvet-macintosh" -->
### Задание 1.

Перейдём к задаче бинарной классификации: будем предсказывать принадлежность цветка к виду `versicolor` против принадлежности ко всем прочим видам. Перекодируйте зависимую переменную так, чтобы цветки вида `versicolor` (y=1) имели метку 1, а прочих видов – метку 0.
<!-- #endregion -->

```python id="balanced-uzbekistan"
# Здесь должен быть ваш код
...
```

<!-- #region id="imported-symphony" -->
### Задание 2.

Будем работать с двумя признаками: `sepal length (cm)` и `sepal width (cm)`. Построим диаграмму рассеяния по тренировочной выборке и убедитесь, что данные линейно не разделимы.
<!-- #endregion -->

```python id="imperial-dealer"
# Здесь должен быть ваш код
X = ...
```

```python id="8be2eb00-a2f4-43e7-8ec0-4e823cac218f"
# делим данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
```

<!-- #region id="a705849e-fc58-416b-9e70-3b7bf9644036" -->
Приведем значения всех входных признаков к одному масштабу. Для этого применим функцию `StandardScaler`. Это преобразование приводит значения каждого признака к нулевому среднему и единичной дисперсии:

$$
X_{new} = \frac{X - \mu}{\sigma}
$$

где, $\mu$ - среднее значение признака

$\sigma$ - стандартное отклонение значений признака
<!-- #endregion -->

```python id="78f784c6-5811-4d69-828c-f46aa70fcc60"
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(X_train, y_train) # считаем \mu и \sigma

# делаем преобразование данных
X_train_ss = ss.transform(X_train)
X_test_ss = ss.transform(X_test)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 430} executionInfo={"elapsed": 530, "status": "ok", "timestamp": 1694978312662, "user": {"displayName": "Sergey Korpachev", "userId": "09181340988160569540"}, "user_tz": -180} id="443a2c2b" outputId="1771433e-79d8-4539-ae7a-80df1afe7a25"
plt.scatter(X_train_ss[:, 0], X_train_ss[:, 1], c=y_train)
plt.show()
```

<!-- #region id="nominated-nightmare" -->
### Задание 3.

Сравним качество для KNN и логрега.
<!-- #endregion -->

```python id="imperial-saskatchewan"
knn = KNeighborsClassifier(n_neighbors=5)
logreg = LogisticRegression()
```

<!-- #region id="cb6f9b02-3f7f-4118-bf07-19b64a12af90" -->
#### Обучение
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 74} executionInfo={"elapsed": 393, "status": "ok", "timestamp": 1694978380567, "user": {"displayName": "Sergey Korpachev", "userId": "09181340988160569540"}, "user_tz": -180} id="b2132597" outputId="f5a6dba3-1993-47a8-d943-31839ca39aad"
# Обучите классификаторы

# Здесь должен быть ваш код
#knn
#logreg
```

```python colab={"base_uri": "https://localhost:8080/", "height": 74} executionInfo={"elapsed": 393, "status": "ok", "timestamp": 1694978380567, "user": {"displayName": "Sergey Korpachev", "userId": "09181340988160569540"}, "user_tz": -180} id="b2132597" outputId="f5a6dba3-1993-47a8-d943-31839ca39aad"
# Решение
# Обучите классификаторы

knn.fit(X_train_ss, y_train)
logreg.fit(X_train_ss, y_train);
```

<!-- #region id="d3e6625e-dbce-4798-9369-66e56165f8cf" -->
#### Прогноз метки класса
<!-- #endregion -->

```python id="62965ba0"
# Получите прогнозы для тестовой выборки

y_test_pred_knn = ...  # Здесь должен быть ваш код
y_test_pred_logreg = ...  # Здесь должен быть ваш код
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 6, "status": "ok", "timestamp": 1694978399424, "user": {"displayName": "Sergey Korpachev", "userId": "09181340988160569540"}, "user_tz": -180} id="b3177ed4-f6a8-4652-bb64-e10825f1842f" outputId="aa04ed47-d14e-4eb4-d608-b5aae0990207"
y_test_pred_knn[:5]
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 6, "status": "ok", "timestamp": 1694978399424, "user": {"displayName": "Sergey Korpachev", "userId": "09181340988160569540"}, "user_tz": -180} id="b3177ed4-f6a8-4652-bb64-e10825f1842f" outputId="aa04ed47-d14e-4eb4-d608-b5aae0990207"
y_test_pred_logreg[:5]
```

<!-- #region id="aa87f46c-c6b4-46c3-822e-9fa4477695b6" -->
#### Прогноз вероятности класса
<!-- #endregion -->

```python id="10378dd9-b25d-4feb-9751-d15af7327212"
# получите прогнозы для тестовой выборки
y_test_proba_knn = ...  # Здесь должен быть ваш код
y_test_proba_logreg = ...  # Здесь должен быть ваш код
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 12, "status": "ok", "timestamp": 1694978420123, "user": {"displayName": "Sergey Korpachev", "userId": "09181340988160569540"}, "user_tz": -180} id="a15978f3-0774-4920-9239-c0c9892fcb0a" outputId="3377eceb-e916-4c1d-cb7a-ee5d637cc1d1"
y_test_proba_knn[:5]
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 12, "status": "ok", "timestamp": 1694978420123, "user": {"displayName": "Sergey Korpachev", "userId": "09181340988160569540"}, "user_tz": -180} id="a15978f3-0774-4920-9239-c0c9892fcb0a" outputId="3377eceb-e916-4c1d-cb7a-ee5d637cc1d1"
y_test_proba_logreg[:5]
```

<!-- #region id="c03f07c6-7fe2-4d10-9a56-bd657d373f3a" -->
#### Метрика качества
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 5, "status": "ok", "timestamp": 1694978421393, "user": {"displayName": "Sergey Korpachev", "userId": "09181340988160569540"}, "user_tz": -180} id="exact-bailey" outputId="cd2f8c20-6138-4944-acab-ea53d39b6d94"
from sklearn.metrics import accuracy_score
print(f'KNN: {accuracy_score(y_test, y_test_pred_knn)}')
print(f'LogReg: {accuracy_score(y_test, y_test_pred_logreg)}')
```

<!-- #region id="945470c3-3cee-440b-b2df-b152e0626621" -->
#### Строим разделяющую поверность
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 716} executionInfo={"elapsed": 55205, "status": "ok", "timestamp": 1694978503375, "user": {"displayName": "Sergey Korpachev", "userId": "09181340988160569540"}, "user_tz": -180} id="04bdac20" outputId="c6a19f97-e090-489d-f9ac-44b6c8cc9da5"
plt.figure(figsize=(12,8))
plot_decision_regions(X_train_ss, y_train, clf=knn, legend=2)
plt.title('Разделяющая поверхность для KNN')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 716} executionInfo={"elapsed": 569, "status": "ok", "timestamp": 1694978503937, "user": {"displayName": "Sergey Korpachev", "userId": "09181340988160569540"}, "user_tz": -180} id="e862265f" outputId="f571debf-4b07-467c-a147-320e674958ef"
plt.figure(figsize=(12,8))
plot_decision_regions(X_train_ss, y_train, clf=logreg, legend=2)
plt.title('Разделяющая поверхность для логрега')
```

<!-- #region id="unsigned-petite" -->
Теперь изучим свойства каждого классификатора по-отдельности. Начнём с логистической регрессии.
<!-- #endregion -->

<!-- #region id="unsigned-petite" -->
### Задание 4.

Обучите три различные логистические регрессии с разным параметром регуляризации $C$.
<!-- #endregion -->

```python id="happy-origin"
# Здесь должен быть ваш код

logreg_1 = ... # C=0.01
logreg_2 = ... # C=0.05
logreg_3 = ... # C=10
```

```python colab={"base_uri": "https://localhost:8080/", "height": 752} executionInfo={"elapsed": 1620, "status": "ok", "timestamp": 1694978793949, "user": {"displayName": "Sergey Korpachev", "userId": "09181340988160569540"}, "user_tz": -180} id="victorian-danger" outputId="f2fe97c4-395c-43e3-c31c-b67d787e5345"
fig, axes = plt.subplots(ncols=3, figsize=(12, 8))
pipes = [logreg_1, logreg_2, logreg_3]

for ind, clf in enumerate(pipes):
    clf.fit(X_train_ss, y_train)
    y_test_pred = clf.predict(X_test_ss)
    score = accuracy_score(y_test, y_test_pred)
    print(f"Правильность при C={clf.get_params()['C']}: ", score)
    fig = plot_decision_regions(X_train_ss, y_train, clf=clf, legend=2, ax=axes[ind])
    fig.set_title(f"C={clf.get_params()['C']}", fontsize=16)
```

<!-- #region id="handy-bolivia" -->
Перейдём к KNN.
<!-- #endregion -->

<!-- #region id="handy-bolivia" -->
### Задание 5.

Обучите три KNN с разным числом соседей.
<!-- #endregion -->

```python id="patent-chess"
# Здесь должен быть ваш код

knn_1 = ... # n_neighbors=1
knn_2 = ... # n_neighbors=5
knn_3 = ... # n_neighbors=50
```

```python colab={"base_uri": "https://localhost:8080/", "height": 752} executionInfo={"elapsed": 176659, "status": "ok", "timestamp": 1694979216181, "user": {"displayName": "Sergey Korpachev", "userId": "09181340988160569540"}, "user_tz": -180} id="local-revolution" outputId="ef16c43e-6beb-4446-e831-cd002e039883"
fig, axes = plt.subplots(ncols=3, figsize=(12, 8))
pipes = [knn_1, knn_2, knn_3]

for ind, clf in enumerate(pipes):
    clf.fit(X_train_ss, y_train)
    y_test_pred = clf.predict(X_test_ss)
    score = accuracy_score(y_test, y_test_pred)
    print(f"Правильность при n_neighbors={clf.get_params()['n_neighbors']}: ", score)
    fig = plot_decision_regions(X_train_ss, y_train, clf=clf, legend=2, ax=axes[ind])
    fig.set_title(f"n_neighbors={clf.get_params()['n_neighbors']}", fontsize=16)
```

<!-- #region id="b947e7eb-22fb-4287-87a7-e0a9093050a0" -->
#### Дополнительные задания
<!-- #endregion -->

<!-- #region id="b947e7eb-22fb-4287-87a7-e0a9093050a0" -->
1. Зачем мы используем `StandardScaler`? 
Что будет, если один из входных признаков умножить на 10^6?
<!-- #endregion -->

```python id="b947e7eb-22fb-4287-87a7-e0a9093050a0"
# Здесь должен быть ваш код
```

2. Найдите оптимальное значение для параметра регуляризации $C$ логистической регрессии

```python id="b947e7eb-22fb-4287-87a7-e0a9093050a0"
# Здесь должен быть ваш код
```

3. Найдите оптимальное количество соседей $K$ в методе ближайших соседей

```python id="b947e7eb-22fb-4287-87a7-e0a9093050a0"
# Здесь должен быть ваш код
```

<!-- #region id="110c7923-494d-4aac-b9de-5cf7a7800efa" -->
## Нелинейные поверхности
<!-- #endregion -->

```python id="87806867-501d-42e0-b8b1-f07b25346932"
from sklearn.datasets import make_circles

X, y = make_circles(n_samples=200, shuffle=True, noise = 0.1, factor=0.1)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 447} executionInfo={"elapsed": 600, "status": "ok", "timestamp": 1694979237125, "user": {"displayName": "Sergey Korpachev", "userId": "09181340988160569540"}, "user_tz": -180} id="d2932909-b194-482f-88f4-8853f70e8552" outputId="930b8ca6-1ca8-4c8c-bdd6-73aa1590d653"
plt.scatter(X[:, 0], X[:, 1], c=y);
```

```python id="48f677a4-fe0b-4d83-8c69-efed2b1eafcc"
# делим данные на обучение и тест
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 74} executionInfo={"elapsed": 7, "status": "ok", "timestamp": 1694979241543, "user": {"displayName": "Sergey Korpachev", "userId": "09181340988160569540"}, "user_tz": -180} id="11b83192-2327-4b88-8dce-6611e011bc20" outputId="a6643ef8-fd5e-447c-a34c-c5dcfc845b33"
# обучаем модель
logreg = LogisticRegression()
logreg.fit(X_train, y_train);
```

```python colab={"base_uri": "https://localhost:8080/", "height": 716} executionInfo={"elapsed": 586, "status": "ok", "timestamp": 1694979243494, "user": {"displayName": "Sergey Korpachev", "userId": "09181340988160569540"}, "user_tz": -180} id="4191ef82-b0e8-4b4a-9ffd-eec3c683bd1b" outputId="6083b823-60aa-4d4d-cc87-7d4af2a09991"
plt.figure(figsize=(12,8))
plot_decision_regions(X_train, y_train, clf=logreg, legend=2)
plt.title('Разделяющая поверхность для логрега');
```

<!-- #region id="e67aaaf9-74d4-40d4-b6ac-ea2475b04c3e" -->
### Добавим новый признак

$$
X_3 = X_1^{2} + X_2^{2}
$$
<!-- #endregion -->

```python id="070d9127-d24f-4031-b370-489f0d3e73db"
X1 = X[:,0]
X2 = X[:,1]
X3 = X1**2+X2**2

X_new = np.c_[X1, X2, X3]
```

```python id="b983936b-de4a-4b2e-8f80-b9b3c3f67782"
# делим данные на обучение и тест
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=123)
```

#### Логистическая регрессия

```python colab={"base_uri": "https://localhost:8080/", "height": 74} executionInfo={"elapsed": 430, "status": "ok", "timestamp": 1694979318290, "user": {"displayName": "Sergey Korpachev", "userId": "09181340988160569540"}, "user_tz": -180} id="77b66f97-85d1-43b0-b5a6-a31e32f0fadd" outputId="27f13d1b-5c5f-4c28-c944-0b1f36f6e83c"
# обучаем модель
logreg = LogisticRegression()
logreg.fit(X_train, y_train);
```

```python colab={"base_uri": "https://localhost:8080/", "height": 699} executionInfo={"elapsed": 997, "status": "ok", "timestamp": 1694979320436, "user": {"displayName": "Sergey Korpachev", "userId": "09181340988160569540"}, "user_tz": -180} id="8be80110-fae1-4287-8b9f-6a5bd964e66f" outputId="aeea4435-dd07-43ed-92fa-de3cb34ed6f3"
# Plot desicion border

x0, x1 = np.meshgrid(np.arange(-1.5, 1.5, 0.01), np.arange(-1.5, 1.5, 0.01))
xx0, xx1 = x0.ravel(), x1.ravel()
X_grid = np.c_[xx0, xx1, xx0**2 + xx1**2]

y_pred = logreg.predict(X_grid)
y_pred = y_pred.reshape(x0.shape)

plt.figure(figsize=(12,8))
plt.contourf(x0, x1, y_pred, levels=1, cmap=plt.cm.seismic, alpha=0.2)
plt.colorbar()
plt.scatter(X[y==0,0], X[y==0, 1], c='b')
plt.scatter(X[y==1,0], X[y==1, 1], c='r');
```

```python colab={"base_uri": "https://localhost:8080/", "height": 693} executionInfo={"elapsed": 1145, "status": "ok", "timestamp": 1694979325559, "user": {"displayName": "Sergey Korpachev", "userId": "09181340988160569540"}, "user_tz": -180} id="b6e4c6ae-5684-4f3a-8c83-1d54115e078f" outputId="48eba74a-3157-44ea-dc8b-2b15630b8e07"
# Plot desicion border

x0, x1 = np.meshgrid(np.arange(-1.5, 1.5, 0.01), np.arange(-1.5, 1.5, 0.01))
xx0, xx1 = x0.ravel(), x1.ravel()
X_grid = np.c_[xx0, xx1, xx0**2 + xx1**2]

y_pred = logreg.predict_proba(X_grid)[:, 1]
y_pred = y_pred.reshape(x0.shape)

plt.figure(figsize=(12,8))
plt.contourf(x0, x1, y_pred, levels=20, cmap=plt.cm.seismic, alpha=0.5)
plt.colorbar()
plt.scatter(X[y==0,0], X[y==0, 1], c='0')
plt.scatter(X[y==1,0], X[y==1, 1], c='0');
```

<!-- #region id="865ba706-6577-4caa-ba93-dc2af360e506" -->
#### Метод ближайших соседей

<!-- #endregion -->

<!-- #region id="25256879-30be-447a-9e8b-10f5298529ec" -->
### Задание 6. Реализуйте аналогичные вычисления для метода ближайших соседей
<!-- #endregion -->

```python id="25256879-30be-447a-9e8b-10f5298529ec"
# Здесь должен быть ваш код
```
