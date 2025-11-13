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

# Метрики качества для регрессии


**Цель работы:** ознакомление с методами оценки качества в задачах классификации.

Продолжительность работы: - 4 часа.

Мягкий дедлайн (10 баллов): 20.11.2025

Жесткий дедлайн (5 баллов): 27.11.2025

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

pd.set_option('display.max_columns', 500)
```

```python
data = pd.read_csv('./data/data_set.csv', delimiter=';')
```

```python
data.head()
```

```python
data.info()
```

## Генерируем данные

```python
# Create the dataset
X = np.linspace(0, 6, 200)[:, np.newaxis]
y = np.sin(X).ravel() + np.sin(6 * X).ravel() + np.random.RandomState(1).normal(0, 0.1, X.shape[0]) + 3
```

```python
X[:5]
```

```python
y[:5]
```

```python
plt.figure(figsize=(9, 6))
plt.scatter(X[:, 0], y)

plt.xlabel('X', size=18)
plt.ylabel('y', size=18)
plt.grid()
plt.show()
```

## Train / Test Split

```python
from sklearn.model_selection import train_test_split

# Split data into train and test samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
```

## Обучаем регрессоры

Будем использовать три модели:
* kNN
* Древо решений
* Линейная регрессия

Мы будем использовать scikit-learn реализацию этих регрессоров. Их описания: [kNN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html), [Decision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html), [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html).

Давайте просто импортируем их и обучим.

```python
# Import kNN regressor
from sklearn.neighbors import KNeighborsRegressor

# Create object of the regressor's class
knn_reg = KNeighborsRegressor(n_neighbors=10)

# Fit the regressor
knn_reg.fit(X_train, y_train)
```

```python
# Import Decision Tree regressor
from sklearn.tree import DecisionTreeRegressor

# Create object of the regressor's class
dt_reg = DecisionTreeRegressor(criterion='squared_error', splitter='best', max_depth=4, 
                            min_samples_split=2, min_samples_leaf=1)

# Fit the regressor
dt_reg.fit(X_train, y_train)
```

```python
# Import Linear Regression regressor
from sklearn.linear_model import LinearRegression

# Create object of the regressor's class
linreg = LinearRegression()

# Fit the regressor
linreg.fit(X_train, y_train)
```

## Прогнозы

```python
# kNN
y_test_knn_reg = knn_reg.predict(X_test)

# DT
y_test_dt_reg = dt_reg.predict(X_test)

# LinReg
y_test_linreg = linreg.predict(X_test)
```

```python
plt.figure(figsize=(15, 6))
plt.scatter(X_test[:, 0], y_test, color='0', label='Truth')

sortd_inds = np.argsort(X_test[:, 0])
plt.plot(X_test[sortd_inds, 0], y_test_knn_reg[sortd_inds], linewidth=3, color='b', label='kNN')
plt.plot(X_test[sortd_inds, 0], y_test_dt_reg[sortd_inds], linewidth=3, color='r', label='DT')
plt.plot(X_test[sortd_inds, 0], y_test_linreg[sortd_inds], linewidth=3, color='g', label='LinReg')

plt.xlabel('X', size=18)
plt.ylabel('y', size=18)
plt.legend(loc='best', fontsize=14)
plt.grid()
plt.show()
```

## Метрики качества

<!-- #region -->
**1. (R)MSE ((Root) Mean Squared Error)**

$$ L(\hat{y}, y) = \frac{1}{N}\sum\limits_{n=1}^N (y_n - \hat{y}_n)^2$$

**2. MAE (Mean Absolute Error)**

$$ L(\hat{y}, y) = \frac{1}{N}\sum\limits_{n=1}^N |y_n - \hat{y}_n|$$

**3. RSE (Relative Squared Error)**

$$ L(\hat{y}, y) = \sqrt\frac{\sum\limits_{n=1}^N (y_n - \hat{y}_n)^2}{\sum\limits_{n=1}^N (y_n - \bar{y})^2}$$

**4. RAE (Relative Absolute Error)**

$$ L(\hat{y}, y) = \frac{\sum\limits_{n=1}^N |y_n - \hat{y}_n|}{\sum\limits_{n=1}^N |y_n - \bar{y}|}$$

**5. MAPE (Mean Absolute Persentage Error)**

$$ L(\hat{y}, y) = \frac{100}{N} \sum\limits_{n=1}^N\left|\frac{ y_n - \hat{y}_n}{y_n}\right|$$


**6. RMSLE (Root Mean Squared Logarithmic Error)**

$$ L(\hat{y}, y) = \sqrt{\frac{1}{N}\sum\limits_{n=1}^N(\log(y_n + 1) - \log(\hat{y}_n + 1))^2}$$
<!-- #endregion -->

```python
from sklearn.metrics import ... # Здесь должен быть ваш код

def regression_quality_metrics_report(y_true, y_pred):
    
    rmse = ... # Здесь должен быть ваш код
    mae = ... # Здесь должен быть ваш код
    rse = ... # Здесь должен быть ваш код
    rae = ... # Здесь должен быть ваш код
    mape = ... # Здесь должен быть ваш код
    rmsle = ... # Здесь должен быть ваш код
    
    return [rmse, mae, rse, rae, mape, rmsle]
```

```python
metrics_report = pd.DataFrame(columns=['RMSE', 'MAE', 'RSE', 'RAE', 'MAPE', 'RMSLE'])

metrics_report.loc['kNN', :] = ... # Здесь должен быть ваш код
metrics_report.loc['DT', :] = ... # Здесь должен быть ваш код
metrics_report.loc['LinReg', :] = ... # Здесь должен быть ваш код

metrics_report
```

**Вопросы**:
- Какой регрессор лучше?
- Как вы можете улучшить качество моделей?
- Как вы можете объяснить поведение регрессоров?
- Что будет, если изменить их гиперпараметры?
