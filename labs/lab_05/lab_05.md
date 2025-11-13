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

# Метрики качества для классификации


**Цель работы:** ознакомление с методами оценки качества в задачах классификации.

Продолжительность работы: - 4 часа.

Мягкий дедлайн (10 баллов): 13.11.2025

Жесткий дедлайн (5 баллов): 20.11.2025

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

pd.set_option('display.max_columns', 500)
```

## Данные

Будем использовать данные [соревнования по анализу данных](http://www.machinelearning.ru/wiki/index.php?title=%D0%97%D0%B0%D0%B4%D0%B0%D1%87%D0%B0_%D0%BF%D1%80%D0%B5%D0%B4%D1%81%D0%BA%D0%B0%D0%B7%D0%B0%D0%BD%D0%B8%D1%8F_%D0%BE%D1%82%D0%BA%D0%BB%D0%B8%D0%BA%D0%B0_%D0%BA%D0%BB%D0%B8%D0%B5%D0%BD%D1%82%D0%BE%D0%B2_%D0%9E%D0%A2%D0%9F_%D0%91%D0%B0%D0%BD%D0%BA%D0%B0_%28%D0%BA%D0%BE%D0%BD%D0%BA%D1%83%D1%80%D1%81%29). 

Одним из способов повышения эффективности взаимодействия банка с клиентами является рассылка предложений о новой услуге не всем клиентам банка, а только определенной части, выбранной исходя из наибольшей склонности реагировать на это предложение.

Задача состоит в том, чтобы предложить алгоритм, который будет оценивать склонность клиента к положительному ответу по его характерному описанию. Это можно интерпретировать как вероятность положительного ответа. Предполагается, что, получив такие оценки для определенного набора клиентов, банк будет адресовать предложение только тем клиентам, вероятность которых выше определенного порога.

### Описание данных 

**ПОЛЕ** &mdash; **ОПИСАНИЕ**
- **AGREEMENT_RK** &mdash; уникальный идентификатор объекта в выборке
- **TARGET** &mdash; целевая переменная:отклик на маркетинговую кампанию (1 - отклик был зарегистрирован, 0 - отклика не было)
- **AGE** &mdash; возраст клиента
- **SOCSTATUS_WORK_FL** &mdash; социальный статус клиента относительно работы (1 - работает, 0 - не работает)
- **SOCSTATUS_PENS_FL** &mdash; социальный статус клиента относительно пенсии (1 - пенсионер, 0 - не пенсионер)
- **GENDER** &mdash; пол клиента
- **CHILD_TOTAL** &mdash; количество детей клиента
- **DEPENDANTS** &mdash; количество иждивенцев клиента
- **EDUCATION** &mdash; образование
- **MARITAL_STATUS** &mdash; семейное положение
- **GEN_INDUSTRY** &mdash; отрасль работы клиента
- **GEN_TITLE** &mdash; должность
- **ORG_TP_STATE** &mdash; форма собственности компании
- **ORG_TP_FCAPITAL** &mdash; отношение к иностранному капиталу
- **JOB_DIR** &mdash; направление деятельности в нутри компании
- **FAMILY_INCOME** &mdash; семейный доход (несколько категорий)
- **PERSONAL_INCOME** &mdash; личный доход клиента (в рублях)
- **REG_ADDRESS_PROVINCE** &mdash; область регистрации клиента
- **FACT_ADDRESS_PROVINCE** &mdash; область фактического пребывания клиента
- **POSTAL_ADDRESS_PROVINCE** &mdash; почтовый адрес область
- **TP_PROVINCE** &mdash; область торговой точки, где клиент брал последний кредит
- **REGION_NM** &mdash; регион РФ
- **REG_FACT_FL** &mdash; адрес регистрации и адрес фактического пребывания клиента совпадают(1 - совпадает, 0 - не совпадает)
- **FACT_POST_FL** &mdash; адрес фактического пребывания клиента и его почтовый адрес совпадают(1 - совпадает, 0 - не совпадает)
- **REG_POST_FL** &mdash; адрес регистрации клиента и его почтовый адрес совпадают(1 - совпадает, 0 - не совпадает)
- **REG_FACT_POST_FL** &mdash; почтовый, фактический и адрес регистрации совпадают (1 - совпадают, 0 - не совпадают)
- **REG_FACT_POST_TP_FL** &mdash; область регистрации, фактического пребывания, почтового адреса и область расположения торговой точки, где клиент брал кредит совпадают (1 - совпадают, 0 - не совпадают)
- **FL_PRESENCE_FL** &mdash; наличие в собственности квартиры (1 - есть, 0 - нет)
- **OWN_AUTO** &mdash; кол-во автомобилей в собственности
- **AUTO_RUS_FL** &mdash; наличие в собственности автомобиля российского производства ( 1 - есть, 0 - нет)
- **HS_PRESENCE_FL** &mdash; наличие в собственности загородного дома (1 - есть, 0 - нет)
- **COT_PRESENCE_FL** &mdash; наличие в собственности котеджа (1 - есть, 0 - нет)
- **GAR_PRESENCE_FL** &mdash; наличие в собственности гаража (1 - есть, 0 - нет)
- **LAND_PRESENCE_FL** &mdash; наличие в собственности земельного участка (1 - есть, 0 - нет)
- **CREDIT** &mdash; сумма последнего кредита клиента (в рублях)
- **TERM** &mdash; срок кредита
- **FST_PAYMENT** &mdash; первоначальный взнос (в рублях)
- **DL_DOCUMENT_FL** &mdash; в анкете клиент указал водительское удостоверение (1 - указал, 0 - не указал)
- **GPF_DOCUMENT_FL** &mdash; в анкете клиен указал ГПФ (1 - указал, 0 - не указал)
- **FACT_LIVING_TERM** &mdash; количество месяцев проживания по месту фактического пребывания
- **WORK_TIME** &mdash; время работы на текущем месте (в месяцах)
- **FACT_PHONE_FL** &mdash; наличие в заявке телефона по фактическому месту пребывания
- **REG_PHONE_FL** &mdash; наличие в заявке телефона по месту регистрации
- **GEN_PHONE_FL** &mdash; наличие в заявке рабочего телефона
- **LOAN_NUM_TOTAL** &mdash; количество ссуд клиента
- **LOAN_NUM_CLOSED** &mdash; количество погашенных ссуд клиента
- **LOAN_NUM_PAYM** &mdash; количество платежей, которые сделал клиент
- **LOAN_DLQ_NUM** &mdash; количество просрочек, допущенных клиентом
- **LOAN_MAX_DLQ** &mdash; номер максимальной просрочки, допущенной клиентом
- **LOAN_AVG_DLQ_AMT** &mdash; средняя сумма просрочки (в рублях)
- **LOAN_MAX_DLQ_AMT** &mdash; максимальная сумма просрочки (в рублях)
- **PREVIOUS_CARD_NUM_UTILIZED** &mdash; количество уже утилизированных карт ( если пусто - 0)


```python
data = pd.read_csv('./data/data_set.csv', delimiter=';')
```

```python
data.head()
```

```python
data.info()
```

## Предобработка #1


Теперь сделаем простую предварительную обработку данных, а именно:

- Удалим &laquo;тяжелые&raquo; признаки `['EDUCATION', 'FACT_ADDRESS_PROVINCE', 'FAMILY_INCOME', 'GEN_INDUSTRY', 'GEN_TITLE', 'JOB_DIR', 'MARITAL_STATUS', 'ORG_TP_FCAPITAL', 'REGION_NM', 'REG_ADDRESS_PROVINCE', 'ORG_TP_STATE', 'POSTAL_ADDRESS_PROVINCE', 'TP_PROVINCE']`
- Заменим запятые на точки в представлении чисел в признаках `['LOAN_AVG_DLQ_AMT', 'LOAN_AVG_DLQ_AMT']`

```python
def preproc(df_input):
    drop_cols = ['EDUCATION', 'FACT_ADDRESS_PROVINCE', 'FAMILY_INCOME', 'GEN_INDUSTRY', 
                 'GEN_TITLE', 'JOB_DIR', 'MARITAL_STATUS', 'ORG_TP_FCAPITAL', 'REGION_NM', 
                 'REG_ADDRESS_PROVINCE', 'ORG_TP_STATE', 'POSTAL_ADDRESS_PROVINCE', 'TP_PROVINCE', 
                 'AGREEMENT_RK']

    # Make a copy of data
    df_temp = df_input.copy()

    # Drop the hard columns
    df_temp = df_temp.drop(drop_cols, axis=1)

    digit_cols = ['LOAN_AVG_DLQ_AMT', 'LOAN_MAX_DLQ_AMT', 'CREDIT', 'FST_PAYMENT', 'PERSONAL_INCOME']
    df_temp[digit_cols] = df_temp[digit_cols].replace(regex={',': '.'}).astype('float64')

    return df_temp
```

```python
data_preproc = preproc(data)
data_preproc.info()
```

## Train / Test Split

```python
label_col = data_preproc.columns == 'TARGET'

# Take all columns that are not TARGET
X = data_preproc.loc[:, ~label_col].values

# Take TARGET column
y = data_preproc.loc[:, label_col].values.flatten()
```

```python
X[:2]
```

```python
y[:10]
```

```python
from sklearn.model_selection import train_test_split

# Split data into train and test samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

## Предобработка #2

Замените пропущенные значения средними значениями признаков. Для этого нужно рассчитать средние значения для каждого входного признака, используя выборку **train**.

```python
# Import imputer
from sklearn.impute import SimpleImputer

# Create object of the class and set up its parameters
imp = SimpleImputer(...) # Здесь должен быть ваш код

# Calculate mean values for each feature
imp.fit(X_train)

# Replace missing values in train and test samples
X_train = imp.transform(X_train)
X_test = imp.transform(X_test)
```

Масштабируйте входные данные с помощью StandardScaler:
$$
X_{new} = \frac{X - \mu}{\sigma}
$$

```python
# Import StandardScaler
from sklearn.preprocessing import StandardScaler

# Create object of the class and set up its parameters
ss = ... # Здесь должен быть ваш код

# Estimate mean and sigma values
ss.fit(X_train)

# Scale train and test samples
X_train = ... # Здесь должен быть ваш код
X_test = ... # Здесь должен быть ваш код
```

<!-- #region -->
## Обучение классификаторов

Cравним три классификатора:
* KNN
* Логистическая регрессия
* Древо решений 

Будем использовать scikit-learn реализацию этих классификаторов. Их описания: [kNN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html), [Decision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html), [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).


[Пример:](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)
<center><img src="./img/clfs.png" width="800"></center>
<!-- #endregion -->

```python
# Import kNN classifier
from sklearn.neighbors import KNeighborsClassifier

# Create object of the classifier's class
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the classifier
knn.fit(X_train, y_train)
```

```python
# Import Decision Tree classifier
from sklearn.tree import DecisionTreeClassifier

# Create object of the classifier's class
dt = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, 
                            min_samples_split=2, min_samples_leaf=10, class_weight=None)

# Fit the classifier
dt.fit(X_train, y_train)
```

```python
# Import Logistic Regression classifier
from sklearn.linear_model import LogisticRegression

# Create object of the classifier's class
logreg = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, class_weight=None)

# Fit the classifier
logreg.fit(X_train, y_train)
```

## Прогнозы


Сделаем прогноз **метки** классов.

```python
# kNN
y_test_knn = knn.predict(X_test)

# Decision Tree
y_test_dt = dt.predict(X_test)

# Logistic Regression
y_test_logreg = logreg.predict(X_test)
```

```python
print("Truth  : ", y_test[:10])
print("kNN    : ", y_test_knn[:10])
print("DT     : ", y_test_dt[:10])
print("LogReg : ", y_test_logreg[:10])
```

Сделаем прогноз **вероятности** положительного ответа клиентов.

```python
# kNN
y_test_proba_knn = ... # Здесь должен быть ваш код

# Decision Tree
y_test_proba_dt = ... # Здесь должен быть ваш код

# Logistic Regression
y_test_proba_logreg = ... # Здесь должен быть ваш код
```

```python
print("Truth  : ", y_test[:10])
print("kNN    : ", y_test_proba_knn[:10])
print("DT     : ", y_test_proba_dt[:10])
print("LogReg : ", y_test_proba_logreg[:10])
```

## Метрики качества для прогнозов меток


Матрица ошибок:

<center><img src='img/binary_conf.png'></center>

* TP (true positive) - currectly predicted positives
* FP (false positive) - incorrectly predicted negatives (1st order error)
* FN (false negative) - incorrectly predicted positives (2nd order error)
* TN (true negative) - currectly predicted negatives
* Pos (Neg) - total number of positives (negatives)

Quality metrics:

* $ \text{Accuracy} = \frac{TP + TN}{Pos+Neg}$
* $ \text{Error rate} = 1 -\text{accuracy}$
* $ \text{Precision} =\frac{TP}{TP + FP}$ 
* $ \text{Recall} =\frac{TP}{TP + FN} = \frac{TP}{Pos}$
* $ \text{F}_\beta \text{-score} = (1 + \beta^2) \cdot \frac{\mathrm{precision} \cdot \mathrm{recall}}{(\beta^2 \cdot \mathrm{precision}) + \mathrm{recall}}$

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def quality_metrics_report(y_true, y_pred):
    
    tp = np.sum( (y_true == 1) * (y_pred == 1) )
    fp = np.sum( (y_true == 0) * (y_pred == 1) )
    fn = np.sum( (y_true == 1) * (y_pred == 0) )
    tn = np.sum( (y_true == 0) * (y_pred == 0) )
    
    accuracy = accuracy_score(y_true, y_pred)
    error_rate = 1 - accuracy
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    return [tp, fp, fn, tn, accuracy, error_rate, precision, recall, f1]
```

```python
metrics_report = pd.DataFrame(columns=['TP', 'FP', 'FN', 'TN', 'Accuracy', 'Error rate', 'Precision', 'Recall', 'F1'])

metrics_report.loc['kNN', :] = quality_metrics_report(y_test, y_test_knn)
metrics_report.loc['DT', :] = quality_metrics_report(y_test, y_test_dt)
metrics_report.loc['LogReg', :] = quality_metrics_report(y_test, y_test_logreg)

metrics_report
```

## Метрики качества на основе вероятностей


### ROC кривая

ROC кривая измеряет насколько хорошо классификатор разделяет два класса.

Пусть $y_{\rm i}$ - истинная метрка и $\hat{y}_{\rm i}$ - прогноз вероятности для $i$-го объекта. 

Число положительных и отрицательных объектов: $\mathcal{I}_{\rm 1} = \{i: y_{\rm i}=1\}$ and $\mathcal{I}_{\rm 0} = \{i: y_{\rm i}=0\}$. 

Для каждого порогового значения вероятности $\tau$ считаем True Positive Rate (TPR) и False Positive Rate (FPR):

\begin{equation}
TPR(\tau) = \frac{1}{I_{\rm 1}} \sum_{i \in \mathcal{I}_{\rm 1}} I[\hat{y}_{\rm i} \ge \tau]
\end{equation}

\begin{equation}
FPR(\tau) = \frac{1}{I_{\rm 0}} \sum_{i \in \mathcal{I}_{\rm 0}} I[\hat{y}_{\rm i} \ge \tau]
\end{equation}

```python
from sklearn.metrics import roc_curve, auc

fpr_knn, tpr_knn, _ = roc_curve(y_test, y_test_proba_knn)
auc_knn = auc(fpr_knn, tpr_knn)

fpr_dt, tpr_dt, _ = roc_curve(y_test, y_test_proba_dt)
auc_dt = auc(fpr_dt, tpr_dt)

fpr_logreg, tpr_logreg, _ = ... # Здесь должен быть ваш код
auc_logreg = ... # Здесь должен быть ваш код
```

```python
plt.figure(figsize=(9, 6))
plt.plot(fpr_knn, tpr_knn, linewidth=3, label='kNN')
plt.plot(fpr_dt, tpr_dt, linewidth=3, label='DT')
plt.plot(fpr_logreg, tpr_logreg, linewidth=3, label='LogReg')

plt.xlabel('FPR', size=18)
plt.ylabel('TPR', size=18)

plt.legend(loc='best', fontsize=14)
plt.grid()
plt.show()

print('kNN ROC AUC    :', auc_knn)
print('DT ROC AUC     :', auc_dt)
print('LogReg ROC AUC :', auc_logreg)
```

### Precision-Recall curve

Аналогично ROC кривой.

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

precision_knn, recall_knn, _ = precision_recall_curve(y_test, y_test_proba_knn)
ap_knn = average_precision_score(y_test, y_test_proba_knn)

precision_dt, recall_dt, _ = precision_recall_curve(y_test, y_test_proba_dt)
ap_dt = average_precision_score(y_test, y_test_proba_dt)

precision_logreg, recall_logreg, _ = precision_recall_curve(y_test, y_test_proba_logreg)
ap_logreg = average_precision_score(y_test, y_test_proba_logreg)
```

```python
plt.figure(figsize=(9, 6))
plt.plot(recall_knn, precision_knn, linewidth=3, label='kNN')
plt.plot(recall_dt, precision_dt, linewidth=3, label='DT')
plt.plot(recall_logreg, precision_logreg, linewidth=3, label='LogReg')

plt.xlabel('Recall', size=18)
plt.ylabel('Precision', size=18)

plt.legend(loc='best', fontsize=14)
plt.grid()
plt.show()

print('kNN AP    :', ap_knn)
print('DT AP     :', ap_dt)
print('LogReg AP :', ap_logreg)
```

**Вопросы**:
* Какой классификатор лучше?
* Как можно улучшить качество моделей?
