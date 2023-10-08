---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

## Логические методы классификации


### Цель работы

изучение принципов построения информационных систем с использованием логических методов классификации.

### Задачи 

 - освоение технологии внедрения алгоритмов на онове решающих списков в приложения;
 - освоение технологии внедрения алгоритмов на онове решающих деревьев в приложения;
 - изучение параметров логической классификации;
 - освоение модификаций логических методов классификации.

### Продолжительность и сроки сдачи

Продолжительность работы: - 4 часа.

Мягкий дедлайн (5 баллов): 10.10.2023

Жесткий дедлайн (2.5 баллов): 24.10.2023


### Теоретические сведения


Перед выполнением лабораторной работы необходимо ознакомиться с базовыми принципами работы со специализированными библиотеками яхыка Python, используя следующие источники: [1]

Перед выполнением лабораторной работы необходимо ознакомиться с базовыми принципами работы с репозитариями  [2, 3]


### Методика и порядок выполнения работы
Перед выполнением индивидуального задания рекомендуется выполнить все пункты учебной задачи.


### Учебная задача

В рамках учебной задачи необходимо произвести построение классификатора на основе логического дерева. 
В качестве набора данных используется набор данных об ирисах Фишера.


#### Подключение библиотек

```python
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
%matplotlib inline
```

#### Получение данных
Необходимо скачать набор данных из одного из репозиториев (необходим только один текстовый файл с данными измерений): 

https://www.kaggle.com/datasets/uciml/iris

http://archive.ics.uci.edu/ml/datasets/Iris.

Заметим, что набор данных Iris доступен в пакете sklearn


#### Загрузка данных
Рассмотрим основные признаки, представленный в наборе.
Загрузим набор данных с использованием `pandas` и выведем признаки набора данных

```python
data_source = "./datasets/iris/iris.data"
data = pd.read_csv(data_source, 
                   delimiter=',', 
                   names=['sepal_length',
                          'sepal_width',
                          'petal_length',
                          'petal_width','answer'],
                   header=None)
data.head(10)
```

```python
X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = data['answer']
```

Для построения дерева классификации воспользуемся специальным классом sklearn.tree.DecisionTreeClassifier. 
Оценими точность модели методом hold-out 

Следует обратить внимание, что если в методе ближайших соседей производилась оптимизация по одному параметру K-количеству ближайших соседей, то при создании модели DecisionTreeClassifier необходимо указать два параметра: максимальную глубину дерева (max_depth) и количество признаков разделения дерева (max_features).

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Подмножества для hold-out
X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.3, random_state=12)

# Обучение модели
tree = DecisionTreeClassifier(max_depth=5, 
                              random_state=21, 
                              max_features=2)
tree.fit(X_train, y_train)

# Получение оценки hold-out
tree_pred = tree.predict(X_holdout)
accur = accuracy_score(y_holdout, tree_pred)
print(accur)
```

Произведем оценку точности модели по методу cross validation, а также сделаем выводы об оптимальном значении параметра
max_depth.

```python
from sklearn.model_selection import cross_val_score

# Значения параметра max_depth
d_list = list(range(1,20))
# Пустой список для хранения значений точности
cv_scores = []
# В цикле проходим все значения K
for d in d_list:
    tree = DecisionTreeClassifier(max_depth=d, 
                                  random_state=21, 
                                  max_features=2)
    scores = cross_val_score(tree, X, y, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

# Вычисляем ошибку (misclassification error)
MSE = [1-x for x in cv_scores]

# Строим график
plt.plot(d_list, MSE)
plt.xlabel('Макс. глубина дерева (max_depth)');
plt.ylabel('Ошибка классификации (MSE)')
plt.show()

# Ищем минимум
d_min = min(MSE)

# Пробуем найти прочие минимумы (если их несколько)
all_d_min = []
for i in range(len(MSE)):
    if MSE[i] <= d_min:
        all_d_min.append(d_list[i])

# печатаем все K, оптимальные для модели
print('Оптимальные значения max_depth: ', all_d_min)
```

Оптимальное значение параметра max_depth модели получено, но в модели присутствует еще один параметр max_features, который был установлен в значение 2 (не изменялся и не оптимизировался). 
Для проведения cross validation по всем параметрам воспользуемся классом GridSearchCV пакета sklearn.model_selection.

```python
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn import tree

dtc = DecisionTreeClassifier(max_depth=10, random_state=21, max_features=2)

tree_params = { 'max_depth': range(1,20), 'max_features': range(1,4) }
tree_grid = GridSearchCV(dtc, tree_params, cv=10, verbose=True, n_jobs=-1)
tree_grid.fit(X, y)

print('\n')
print('Лучшее сочетание параметров: ', tree_grid.best_params_)
print('Лучшие баллы cross validation: ', tree_grid.best_score_)

```

Поясните вывод данного фрагмента. 
Поясните значение таких величин как fold, candidate, fit. 
Какие значения принимают данные величины в данном коде и почему?


Следует обратить внимание, что в результате оценки оптимальных параметров, фактически, было построено оптимальное дерево классификации.
Доступ к данному дереву производится через поле best_estimator_ класса GridSearchCV. 


Визуализировать дерево решений можно, используя функцию `export_graphviz` из модуля `tree`. 
Она записываеь файл в формате .dot, который является форматом текстового файла, предназначенным для описания графиков.
При этом можно задать цвет узлам, чтобы выделить класс, набравший большинство в каждом узле, и передать имена классов и признаков, чтобы дерево было правильно размечено:

```python
from sklearn.tree import export_graphviz
# Генерируем графическое представление дерева
tree.export_graphviz(tree_grid.best_estimator_, 
                     feature_names=X.columns,
                     class_names=y.unique(),
                     out_file='./img/iris_tree.dot',  
                     filled=True, rounded=True);

```

Отрисуем дерево решений

```python
import graphviz

with open('./img/iris_tree.dot') as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)
```

Поясните значения переменных в узлах полученного дерева: gini, samples, value.


 В заключении построим еще одну визуализацию процесса логической классификации - покажем решающие границы модели классификации

```python
plot_markers = ['r*', 'g^', 'bo']
answers = y.unique()

# Создаем подграфики для каждой пары признаков
f, places = plt.subplots(4, 4, figsize=(16,16))

fmin = X.min()-0.5
fmax = X.max()+0.5
plot_step = 0.02  

# Обходим все subplot
for i in range(0,4):
    for j in range(0,4):

        # Строим решающие границы 
        if(i != j):
            xx, yy = np.meshgrid(np.arange(fmin[i], fmax[i], plot_step),
                               np.arange(fmin[j], fmax[j], plot_step))
            model = DecisionTreeClassifier(max_depth=3, random_state=21, max_features=2)
            model.fit(X.iloc[:, [i,j]].values, y)
            p = model.predict(np.c_[xx.ravel(), yy.ravel()])
            p = p.reshape(xx.shape)
            p[p==answers[0]] = 0
            p[p==answers[1]] = 1
            p[p==answers[2]] = 2
            xx = xx.astype(np.float32)
            yy = yy.astype(np.float32)
            p = p.astype(np.float32)
            places[i,j].contourf(xx, yy, p, cmap='Pastel1') 
      
        # Обход всех классов
        for id_answer in range(len(answers)):
            idx = np.where(y == answers[id_answer])
            if i==j:
                places[i, j].hist(X.iloc[idx].iloc[:,i],
                                  color=plot_markers[id_answer][0],
                                 histtype = 'step')
            else:
                places[i, j].plot(X.iloc[idx].iloc[:,i], X.iloc[idx].iloc[:,j], 
                                  plot_markers[id_answer], 
                                  label=answers[id_answer], markersize=6)
        
        if j==0:
            places[i, j].set_ylabel(X.columns[j])
        
        if i==3:
            places[i, j].set_xlabel(X.columns[i])
        
    
```

#### Использование модели
Оптимальные параметры определены, можно обучить модель и использовать ее для классификации

```python
# Построим модель для оптимального дерева
# max_features = 2, max_depth = 3

dtc = DecisionTreeClassifier(max_depth=3, 
                             random_state=21, 
                             max_features=2)
dtc.fit(X.values, y.values)


# Использование классификатора
# Объявление признаков объекта

sepal_length = float(input('Введите длину чашелистика: '))
sepal_width = float(input('Введите ширину чашелистика: '))
petal_length = float(input('Введите длину лепестка: '))
petal_width = float(input('Введите ширину лепестка: '))
X_new = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Получение ответа для нового объекта
target = dtc.predict(X_new)[0]


print('\nДанный цветок относится к виду: ',target)


```

<!-- #region jp-MarkdownHeadingCollapsed=true -->
### Важные замечания

При выборе набора данных следует отдавать предпочтение тем наборам, которые имеют следующие характеристики: содержат не более 5 признаков на объект; все признаки – числовые; желательно отсутствие пропусков в данных.
<!-- #endregion -->

### Индивидуальное задание

1. Подберите набор данных на ресурсах [2, 3] и согласуйте свой выбор с преподавателем и другими студентами группы, так
как работа над одинаковыми наборами данных недопустима.

2. Выполните построение модели классификации на основе дерева классификации. В ходе решения задачи необходимо решить следующие подзадачи:
 
 - Построение логического классификатора с заданием max_depth (максимальной глубины) и max_features (максимального количества признаков) пользователем (установить любые); визуализация дерева решений для выбранных исследователем параметров; 
 - Вычисление оценки cross validation (MSE) для различнх значений max_depth (построить график зависимости);
 - Вычисление оценки cross validation (MSE) для различнх значений max_features (построить график зависимости);
 - Вычислите оптимальные значения max_depth и max_features. Обоснуйте свой выбор. Продемонстрируйте использование полученного классификатора.

3. Выведите оптимальное дерево решений

4. Выведите решающие границы полученной модели.


### Содержание отчета и его форма

Отчет по лабораторной работе должен содержать:

1. Номер и название лабораторной работы; задачи лабораторной работы.

2. Реализация каждого пункта подраздела «Индивидуальное задание» с приведением исходного кода программы, диаграмм и графиков для визуализации данных.

3. Ответы на контрольные вопросы.

4. Листинг программного кода с комментариями, показывающие порядок выполнения лабораторной работы, и результаты, полученные в ходе её выполнения.


### Контрольные вопросы

1. Поясните принцип построения дерева решений.
2. Укажите статистическое определение информативности.
3. Поясните энтропийное определение информативности.
4. Что такое многоклассовая информативность? Для чего она применяется?
5. Поясните назначение и алгоритм бинаризации количественных признаков.
6. Поясните порядок поиска закономерностей в форме конъюнкций.


### Список литературы

1. Дж. Плас: Python для сложных задач. Наука о данных и машинное обучение. Питер.,2018, 576 с.

2. [Репозиторий наборов данных для машинного обучения (Центр машинного обучения и интеллектуальных систем)](https://archive.ics.uci.edu/datasets)

3. [Репозиторий наборов данных для машинного обучения (Kaggle)](https://www.kaggle.com/datasets/)
