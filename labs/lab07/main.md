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

# Предсказание пола клиента по транзакциям


## Описание задачи

Одним из самых ценных источников информации о клиенте являются данные о банковских транзакциях. В этом задании предлагается ответить на вопрос: возможно ли **предсказать пол клиента**, используя сведения о поступлениях и тратах по банковской карте? И если возможно, то какая точность такого предсказания?

Оригинальная постановка задачи: https://www.kaggle.com/competitions/python-and-analyze-data-final-project

### Формальная постановка задачи

Необходимо предсказать вероятность пола "1" для каждого "customerid", который присутствует в файле gendertestkagglesample_submission.csv.

Для понимания представленных данных будет полезна следующая [статья](https://www.banki.ru/wikibank/mcc-kod/)

В роли метрики выступает [ROC AUC](https://ru.wikipedia.org/wiki/ROC-%D0%BA%D1%80%D0%B8%D0%B2%D0%B0%D1%8F), который и нужно будет оптимизировать.¶

Полученная модель должна иметь ROC AUC на тестовой выборке (gender_test.csv) не менее 80%

<!-- #region -->
## Описание данных

**transactions.csv** - таблица содержит историю транзакций клиентов банка за один год и три месяца.

 - customer_id - идентификатор клиента
 
 - tr_datetime - день и время совершения транзакции (дни нумеруются с начала данных)
 
 - mcc_code - mcc-код транзакции
 
 - tr_type - тип транзакции
 
 - amount - сумма транзакции в условных единицах; со знаком "+" — начисление средств клиенту (приходная транзакция), "-" — списание средств (расходная транзакция)
 
 - term_id - идентификатор терминала


**gender_train.csv** - таблица содержит информацию по полу для части клиентов, для которых он известен.

 - customer_id - идентификатор клиента
 
 - gender - пол клиента

**gender_test_kaggle_sample_submission.csv** - пример файла для загрузки решения на Kaggle. Структура таблицы аналогична **gender_train.csv**


**tr_mcc_codes.csv** - таблица содержит описание mcc-кодов транзакций.

 - mcc_code - mcc-код транзакции
 - mcc_description - описание mcc-кода транзакции

**tr_types.csv** - таблица содержит описание типов транзакций.

 - tr_type - тип транзакции
 - tr_description - описание типа транзакции

**gender_pred_sample.csv** - пример файла решения.

 - customer_id - идентификатор клиента
 - probability - вероятность принадлежности к полу "1"

<!-- #endregion -->

```python
import pandas as pd
import numpy as np

import xgboost as xgb
import re
import matplotlib.pyplot as plt

#from tqdm.notebook import tqdm_notebook
from warnings import filterwarnings

%matplotlib inline
filterwarnings('ignore')
```

## Подготовка данных

```python
# Считываем данные
tr_mcc_codes = pd.read_csv('./data/tr_mcc_codes.csv', sep=';', index_col='mcc_code')
tr_types = pd.read_csv('./data/tr_types.csv', sep=';', index_col='tr_type')

transactions = pd.read_csv('./data/transactions.csv', index_col='customer_id')
transactions.describe()
```

```python
gender_train = pd.read_csv('./data/gender_train.csv', index_col='customer_id')
gender_test = pd.DataFrame(columns=['gender'], index = list(set(transactions.index) - set(gender_train.index)))
# gender_test = pd.read_csv('./data/gender_test_kaggle_sample_submission.csv', index_col='customer_id')

transactions_train = transactions.join(gender_train, how='inner')
transactions_test = transactions.join(gender_test, how='inner')
```

```python
print(f'Всего уникальных клиентов: {transactions.index.nunique()}')
print(f'Всего уникальных клиентов с известным полом (train): {transactions_train.index.nunique()}')
print(f'Всего уникальных клиентов с неизвестным полом (test): {transactions_test.index.nunique()}')
```

## Формирование признаков

```python
# Добавим дополнительные признаки по каждому пользователю в модель.
# Для этого будем анализировать дни недели, часы и состояние дня/ночи.
for df in [transactions_train, transactions_test]:
    df['weekday'] = df['tr_datetime'].str.split().apply(lambda x: int(x[0]) % 7)
    df['hour'] = df['tr_datetime'].apply(lambda x: re.search(' \d*', x).group(0)).astype(int)
    df['isday'] = df['hour'].between(6, 22).astype(int)

transactions_train.sample
```

```python
def features_creation(x): 
    '''Формирование признаков по каждому пользователю'''
    
    features = []

    # ВременнЫе признаки
    features.append(pd.Series(x['weekday'].value_counts(normalize=True).add_prefix('weekday_')))
    features.append(pd.Series(x['hour'].value_counts(normalize=True).add_prefix('hour_')))
    features.append(pd.Series(x['isday'].value_counts(normalize=True).add_prefix('isday_')))
    
    # Стандартные агрегации, посчитанные на расходах и приходах клиента: 
    # минимум, максимум, среднее, медиана, среднеквадратичное отклонение, количество
    features.append(pd.Series(x[x['amount']>0]['amount'].agg(['min', 'max', 'mean', 'median', 'std', 'count'])\
                                                        .add_prefix('positive_transactions_')))
    features.append(pd.Series(x[x['amount']<0]['amount'].agg(['min', 'max', 'mean', 'median', 'std', 'count'])\
                                                        .add_prefix('negative_transactions_')))

    # Типы транзакций 
    #features.append(pd.Series(x['mcc_code'].value_counts(normalize=True).add_prefix('mcc_code_')))
    # Типы mcc кодов
    #features.append(pd.Series(x['tr_type'].value_counts(normalize=True).add_prefix('tr_type_')))

    # ДОПОЛНИТЕЛЬНЫЕ ПАРАМЕТРЫ
    # ....
    
    return pd.concat(features)
```

```python
data_train = transactions_train.groupby(transactions_train.index).apply(features_creation).unstack(-1)
data_test = transactions_test.groupby(transactions_test.index).apply(features_creation).unstack(-1)
```

## Построение модели

Функции, которыми можно пользоваться для построения классификатора, оценки его результатов и построение прогноза для тестовой части пользователей

```python
def cv_score(params, train, y_true):
    '''Cross-validation score (среднее значение метрики ROC AUC на тренировочных данных)'''
    cv_res=xgb.cv(params, xgb.DMatrix(train, y_true),
                  early_stopping_rounds=10, maximize=True, 
                  num_boost_round=10000, nfold=5, stratified=True)
    index_argmax = cv_res['test-auc-mean'].argmax()
    print(f'Cross-validation, ROC AUC: {cv_res.loc[index_argmax]['test-auc-mean']:.3f}+-{cv_res.loc[index_argmax]['test-auc-std']:.3f}')
    print(f'Trees: {index_argmax}')
```

```python
def fit_predict(params, num_trees, train, test, target):
    '''Построение модели + возврат результатов классификации тестовых пользователей'''
    params['learning_rate'] = params['eta']
    clf = xgb.train(params, xgb.DMatrix(train.values, target, feature_names=list(train.columns)), 
                    num_boost_round=num_trees, maximize=True)
    y_pred = clf.predict(xgb.DMatrix(test.values, feature_names=list(train.columns)))
    submission = pd.DataFrame(index=test.index, data=y_pred, columns=['probability'])
    return clf, submission
```

```python
def draw_feature_importances(clf, top_k=10):
    '''Отрисовка важности переменных. Важность переменной - количество разбиений выборки, 
    в которых участвует данная переменная. Чем больше - тем она, вероятно, лучше 
    '''
    plt.figure(figsize=(10, 10))
    
    importances = dict(sorted(clf.get_score().items(), key=lambda x: x[1])[-top_k:])
    y_pos = np.arange(len(importances))
    
    plt.barh(y_pos, list(importances.values()), align='center', color='green')
    plt.yticks(y_pos, importances.keys(), fontsize=12)
    plt.xticks(fontsize=12)
    plt.xlabel('Feature importance', fontsize=15)
    plt.title('Features importances, Sberbank Gender Prediction', fontsize=18)
    plt.ylim(-0.5, len(importances) - 0.5)
    plt.show()
```

```python
# Стандартные параметры модели
# !!! Здесь можно и НУЖНО экспериментировать

params = {
    'eta': 0.1,
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    
    'gamma': 0,
    'lambda': 0,
    'alpha': 0,
    'min_child_weight': 0,
    
    'eval_metric': 'auc',
    'objective': 'binary:logistic' ,
    'booster': 'gbtree',
    'njobs': -1,
    'tree_method': 'approx'
}
```

### Построение решения

```python
target = data_train.join(gender_train, how='inner')['gender']
cv_score(params, data_train, target)
```

```python
### Число деревьев для XGBoost имеет смысл выставлять по результатам на кросс-валидации 
clf, submission = fit_predict(params, 180, data_train, data_test, target)
```

```python
draw_feature_importances(clf, 10)
```

```python
# В итоге можем отправить полученное решение на платформу Kaggle. 
# Для этого выгрузим его в *.csv - файл, после чего полученный файл можем загружать в качестве ответа.
submission.to_csv('./data/submission.csv')
```

## Дополнительные задания


Задание 1. Связать номер дня с календарем. 

Т.е. определить дату начала наблюдений

```python
#  your code
```

Задание 2. Декодировать суммы поступлений и списаний. 

Подсказка: сумма транзакции в условных единицах (amount) получена из реальной суммы транзакции умноженной на некий "секретный" коэффициент. 

```python
#  your code
```

```python
При разведочном анализе желательно построить гистограмы распределения транзакций по дням недели, по часам, по дням ... 
```
