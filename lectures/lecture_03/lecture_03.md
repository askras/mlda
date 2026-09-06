---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.3
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region editable=true slideshow={"slide_type": "slide"} -->
# Лекция 3: Подготовка данных

Машинное обучение и анализ данных

МГТУ им. Н.Э. Баумана

Красников Александр Сергеевич

2026-2027

<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "slide"} -->
## Фильтрация выбросов (Outlier Detection)

**Выброс (Outlier) или Аномалия (Anomaly)** — это нетипичный объект в выборке, который существенно отклоняется от основного распределения данных.

**Проблема:**
Выбросы могут существенно **исказить процесс обучения** модели, привести к неверным выводам и снизить ее точность и устойчивость.

**Аналогия:**
*   **Типичные объекты** — это основная стая птиц.
*   **Выбросы** — это несколько птиц, летящих совсем в другом направлении.
<!-- #endregion -->

```python editable=true slideshow={"slide_type": "slide"}
import matplotlib.pyplot as plt
import numpy as np
import random

# Настройка стиля
plt.style.use('default')
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 12

# Создание данных - основная группа точек
np.random.seed(42)  # для воспроизводимости результатов
n_points = 100

# Основное нормальное распределение
x_main = np.random.normal(50, 10, n_points)
y_main = np.random.normal(50, 10, n_points)

# Создание выбросов - точек на большом расстоянии
n_outliers = 8
outliers_x = []
outliers_y = []

# Разные типы выбросов
for i in range(n_outliers):
    if i % 4 == 0:
        # Выбросы в правом верхнем углу
        outliers_x.append(random.uniform(85, 95))
        outliers_y.append(random.uniform(85, 95))
    elif i % 4 == 1:
        # Выбросы в левом нижнем углу
        outliers_x.append(random.uniform(5, 15))
        outliers_y.append(random.uniform(5, 15))
    elif i % 4 == 2:
        # Выбросы в правом нижнем углу
        outliers_x.append(random.uniform(85, 95))
        outliers_y.append(random.uniform(5, 15))
    else:
        # Выбросы в левом верхнем углу
        outliers_x.append(random.uniform(5, 15))
        outliers_y.append(random.uniform(85, 95))

# Создание графика
fig, ax = plt.subplots(figsize=(12, 8))

# Построение основной группы точек
main_scatter = ax.scatter(x_main, y_main, 
                         alpha=0.7, 
                         s=60, 
                         color='steelblue',
                         edgecolors='white',
                         linewidth=0.5,
                         label='Типичные объекты')

# Построение выбросов
outlier_scatter = ax.scatter(outliers_x, outliers_y, 
                            alpha=0.8, 
                            s=80, 
                            color='red',
                            edgecolors='darkred',
                            linewidth=1,
                            marker='X',
                            label='Выбросы (Outliers)')

# Настройка внешнего вида
ax.set_xlabel('Признак X', fontsize=14, fontweight='bold')
ax.set_ylabel('Признак Y', fontsize=14, fontweight='bold')
ax.set_title('Визуализация выбросов в данных\n', fontsize=16, fontweight='bold')

# Добавление сетки для лучшей читаемости
ax.grid(True, alpha=0.3, linestyle='--')

# Добавление легенды
ax.legend(loc='upper left', fontsize=12, framealpha=0.9)

# Установка равных масштабов по осям
ax.set_aspect('equal')
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

# Добавление аннотаций для пояснения
ax.annotate('Основное распределение\nтипичных объектов', 
            xy=(50, 50), 
            xytext=(30, 70),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5),
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

ax.annotate('Объекты-выбросы\nдалеко от основного\nраспределения', 
            xy=(outliers_x[0], outliers_y[0]), 
            xytext=(70, 90),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            fontsize=11,
            color='darkred',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))

plt.tight_layout()
plt.show()

# Дополнительная информация в консоли
print(f"Всего точек: {len(x_main) + len(outliers_x)}")
print(f"Типичные объекты: {len(x_main)}")
print(f"Выбросы: {len(outliers_x)}")
```

<!-- #region editable=true slideshow={"slide_type": "slide"} -->
### Причины появления выбросов

1.  **Ошибки измерения или ввода данных**
    *   *Пример:* Операционист неверно занес данные (например, рост человека 250 см вместо 175 см).
    *   *Пример:* Сбой сенсора или оборудования.

2.  **Реальные, но нетипичные события**
    *   *Пример:* В интернет-трафике — активность ботов, а не реальных пользователей.
    *   *Пример:* В финансах — мошеннические операции.
    *   *Пример:* В медицине — редкое генетическое заболевание.
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "slide"} -->
### Зачем фильтровать выбросы?

**Цель фильтрации:** Настроить модель так, чтобы она была **устойчивее** и лучше обрабатывала именно **типичные случаи**.

**Последствия игнорирования выбросов:**
*   **Занижение/завышение** прогнозов модели.
*   **Неадекватные** веса признаков.
*   **Завышенная** ошибка модели.
*   **Низкая** обобщающая способность на новых данных.
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "slide"} -->
### Подходы к обнаружению выбросов

**1. Одномерный анализ (Univariate)**
*   Анализ распределения **каждого признака в отдельности**.
*   **Методы:** Правило 3-х сигм (Z-score), межквартильный размах (IQR), визуализация (boxplot, гистограмма).
*   **Плюсы:** Простота и скорость.
*   **Минусы:** Не учитывает взаимосвязи между признаками.

**2. Многомерный анализ (Multivariate)**
*   Анализ **всего вектора признаков** объекта одновременно.
*   **Методы:** Isolation Forest, Local Outlier Factor (LOF), DBSCAN, Elliptic Envelope.
*   **Плюсы:** Выявляет более сложные, неочевидные аномалии.
*   **Минусы:** Вычислительно сложнее, требует большего понимания данных.
<!-- #endregion -->

```python editable=true slideshow={"slide_type": "slide"}
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.covariance import EllipticEnvelope
import matplotlib.patches as patches

# Настройка стиля
plt.style.use('default')
plt.rcParams['figure.figsize'] = [12, 5]
plt.rcParams['font.size'] = 12

# Создание данных с выбросами
np.random.seed(42)

# Данные для одномерного анализа (боксплот)
data_normal = np.random.normal(50, 10, 100)
data_outliers = np.concatenate([data_normal, [5, 95, 100, 3, 102]])

# Данные для многомерного анализа
X, y = make_blobs(n_samples=100, centers=1, cluster_std=1.0, random_state=42)

# Масштабируем и сдвигаем данные
X[:, 0] = X[:, 0] * 10 + 70  
X[:, 1] = X[:, 1] * 5

# Добавляем выбросы
outliers = np.array([
    [80, 30],
    [75, 75],
    [25, 80],
    [20, 20],
    [90, 85]
])
X_with_outliers = np.vstack([X, outliers])
y_with_outliers = np.concatenate([np.zeros(len(X)), np.ones(len(outliers))])  # Метки: 0 - нормальные, 1 - выбросы

# Обучение Elliptic Envelope для обнаружения выбросов
ellipse = EllipticEnvelope(contamination=0.05, random_state=42)
ellipse.fit(X)
xx, yy = np.meshgrid(np.linspace(0, 100, 100), np.linspace(0, 100, 100))
Z = ellipse.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Создание фигуры с двумя подграфиками
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Левый график: Боксплот (одномерный анализ)
boxplot = ax1.boxplot([data_normal, data_outliers], 
                     patch_artist=True,
                     tick_labels=['Без выбросов', 'С выбросами'],
                     widths=0.6)

# Настройка внешнего вида боксплота
colors = ['lightblue', 'lightcoral']
for patch, color in zip(boxplot['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

for whisker in boxplot['whiskers']:
    whisker.set(color='black', linewidth=1.5, linestyle='-')

for cap in boxplot['caps']:
    cap.set(color='black', linewidth=1.5)

for median in boxplot['medians']:
    median.set(color='red', linewidth=2)

for flier in boxplot['fliers']:
    flier.set(marker='o', color='red', alpha=0.8, markersize=8)

ax1.set_title('Одномерный анализ: Боксплот\n', fontsize=14, fontweight='bold')
ax1.set_ylabel('Значение признака', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_ylim(0, 110)

# Добавление аннотаций для боксплота
ax1.annotate('Выбросы', xy=(2, 102), xytext=(2.2, 105),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            fontsize=11, color='darkred', ha='center')

# Правый график: Scatter plot с эллипсом (многомерный анализ)
# Рисуем контур эллипса
contour = ax2.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkgreen', linestyles='--')

# Рисуем нормальные точки
normal_points = ax2.scatter(X[:, 0], X[:, 1], 
                           alpha=0.7, 
                           s=50, 
                           color='green',
                           edgecolors='darkblue',
                           linewidth=0.5,
                           marker='o',
                           label='Нормальные объекты')

# Рисуем выбросы
outlier_points = ax2.scatter(outliers[:, 0], outliers[:, 1], 
                            alpha=0.8, 
                            s=80, 
                            color='red',
                            edgecolors='darkred',
                            linewidth=1,
                            marker='X',
                            label='Выбросы')

ax2.set_title('Многомерный анализ: Scatter plot с эллипсом\n', fontsize=14, fontweight='bold')
ax2.set_xlabel('Признак X', fontsize=12, fontweight='bold')
ax2.set_ylabel('Признак Y', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xlim(0, 100)
ax2.set_ylim(0, 100)
ax2.legend(loc='upper right', fontsize=11, framealpha=0.9)

# Добавление аннотаций для scatter plot
ax2.annotate('Область типичных\nзначений (эллипс)', 
            xy=(50, 50), 
            xytext=(30, 70),
            arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
            fontsize=11,
            color='darkgreen',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8))

ax2.annotate('Многомерные\nвыбросы', 
            xy=(80, 30), 
            xytext=(65, 15),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            fontsize=11,
            color='darkred',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))

# Общий заголовок
plt.suptitle('Сравнение методов обнаружения выбросов: одномерный vs многомерный анализ', 
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()
plt.show()

# Дополнительная информация в консоли
print("Одномерный анализ (боксплот):")
print(f"- Всего точек: {len(data_outliers)}")
print(f"- Выбросы обнаружены: {len(data_outliers) - len(data_normal)}")
print("\nМногомерный анализ (scatter plot):")
print(f"- Всего точек: {len(X_with_outliers)}")
print(f"- Выбросы обнаружены: {len(outliers)}")
print(f"- Эллипс охватывает примерно 95% нормальных данных")
```

<!-- #region editable=true slideshow={"slide_type": "slide"} -->
### Простые методы одномерной фильтрации

**Межквартильный размах (IQR)**

Формула:

`Нижняя граница = Q1 - 1.5 * IQR`

`Верхняя граница = Q3 + 1.5 * IQR`

*где Q1 — первый квартиль (25-й перцентиль), Q3 — третий квартиль (75-й перцентиль), IQR = Q3 - Q1*

Все объекты за этими границами считаются потенциальными выбросами.
<!-- #endregion -->

```python editable=true slideshow={"slide_type": "slide"}
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# Настройка стиля
plt.style.use('default')
plt.rcParams['figure.figsize'] = [10, 8]
plt.rcParams['font.size'] = 12

# Создание данных с выбросами
np.random.seed(42)
data = np.concatenate([
    np.random.normal(50, 10, 100),  # Основное распределение
    [5, 95, 100, 3, 102, 110, 2]    # Выбросы
])

# Вычисление статистик
q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)
iqr = q3 - q1
median = np.median(data)
lower_whisker = q1 - 1.5 * iqr
upper_whisker = q3 + 1.5 * iqr

# Создание фигуры
fig, ax = plt.subplots(figsize=(12, 10))

# Создание boxplot
boxplot = ax.boxplot(data, 
                     orientation='horizontal', 
                     whis=1.5, 
                     patch_artist=True, 
                     widths=1,
                     manage_ticks=True,)

# Настройка внешнего вида boxplot
colors = {
    'box': 'lightblue',
    'median': 'red',
    'whiskers': 'black',
    'caps': 'black',
    'fliers': 'red'
}

# Изменение цветов элементов boxplot
boxplot['boxes'][0].set_facecolor(colors['box'])
boxplot['boxes'][0].set_alpha(0.7)
boxplot['boxes'][0].set_edgecolor('black')

for whisker in boxplot['whiskers']:
    whisker.set(color=colors['whiskers'], linewidth=2, linestyle='-')

for cap in boxplot['caps']:
    cap.set(color=colors['caps'], linewidth=2)

for median_line in boxplot['medians']:
    median_line.set(color=colors['median'], linewidth=3)

for flier in boxplot['fliers']:
    flier.set(marker='o', color=colors['fliers'], alpha=0.8, markersize=8, markeredgecolor='darkred')

# Добавление линий и аннотаций для Q1, Q3, IQR
# Линия и аннотация для Q1
ax.axvline(x=q1, color='green', linestyle='--', alpha=0.7, linewidth=2)
ax.annotate(f'Q1 = {q1:.1f}', xy=(q1, 1.15), xytext=(q1, 1.25),
           arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
           fontsize=12, color='darkgreen', ha='center',
           bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", ec="green", alpha=0.8))

# Линия и аннотация для Q3
ax.axvline(x=q3, color='blue', linestyle='--', alpha=0.7, linewidth=2)
ax.annotate(f'Q3 = {q3:.1f}', xy=(q3, 1.15), xytext=(q3, 1.25),
           arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
           fontsize=12, color='darkblue', ha='center',
           bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", ec="blue", alpha=0.8))

# Линия и аннотация для медианы
ax.axvline(x=median, color='red', linestyle='--', alpha=0.7, linewidth=2)
ax.annotate(f'Медиана = {median:.1f}', xy=(median, 0.85), xytext=(median, 0.75),
           arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
           fontsize=12, color='darkred', ha='center',
           bbox=dict(boxstyle="round,pad=0.3", fc="lightcoral", ec="red", alpha=0.8))

# Подсветка области IQR
ax.axvspan(q1, q3, alpha=0.2, color='orange', ymin=0.4, ymax=0.6)
ax.annotate(f'IQR = Q3 - Q1 = {iqr:.1f}', xy=((q1+q3)/2, 0.5), xytext=((q1+q3)/2, 0.65),
           fontsize=14, color='darkorange', ha='center', fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", fc="wheat", ec="orange", alpha=0.9))

# Аннотации для выбросов
outliers = [x for x in data if x < lower_whisker or x > upper_whisker]
for i, outlier in enumerate(outliers[:3]):  # Аннотируем первые 3 выброса
    ax.annotate('Выброс', xy=(outlier, 1.0), xytext=(outlier, 1.1),
               arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
               fontsize=11, color='darkred', ha='center',
               bbox=dict(boxstyle="round,pad=0.3", fc="lightcoral", ec="red", alpha=0.8))

# Линии и аннотации для усов
#ax.axvline(x=lower_whisker, color='purple', linestyle=':', alpha=0.7, linewidth=2)
#ax.axvline(x=upper_whisker, color='purple', linestyle=':', alpha=0.7, linewidth=2)

ax.annotate(f'Нижний ус\nQ1 - 1.5×IQR = {lower_whisker:.1f}', 
           xy=(lower_whisker, 1.0), xytext=(lower_whisker-15, 1.2),
           arrowprops=dict(arrowstyle='->', color='purple', lw=1.5),
           fontsize=11, color='darkviolet', ha='center',
           bbox=dict(boxstyle="round,pad=0.3", fc="plum", ec="purple", alpha=0.8))

ax.annotate(f'Верхний ус\nQ3 + 1.5×IQR = {upper_whisker:.1f}', 
           xy=(upper_whisker, 1.0), xytext=(upper_whisker+15, 1.2),
           arrowprops=dict(arrowstyle='->', color='purple', lw=1.5),
           fontsize=11, color='darkviolet', ha='center',
           bbox=dict(boxstyle="round,pad=0.3", fc="plum", ec="purple", alpha=0.8))

# Настройка осей и заголовка
ax.set_xlabel('Значения', fontsize=14, fontweight='bold')
ax.set_title('Диаграмма Boxplot с аннотациями\nQ1, Q3, IQR и выбросы', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_yticks([1])
ax.set_yticklabels(['Данные'])
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(-5, 120)
#ax.set_xticks([1,10, lower_whisker, median, upper_whisker, q1, q3])

# Добавление легенды
legend_elements = [
    plt.Line2D([0], [0], color='green', linestyle='--', linewidth=2, label='Q1 (25-й перцентиль)'),
    plt.Line2D([0], [0], color='blue', linestyle='--', linewidth=2, label='Q3 (75-й перцентиль)'),
    plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Медиана'),
    plt.Line2D([0], [0], color='purple', linestyle=':', linewidth=2, label='Усы (1.5×IQR)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Выбросы'),
    patches.Patch(facecolor='orange', alpha=0.3, label='IQR (Interquartile Range)')
]

ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9)

# Добавление информационной панели
info_text = f'''Статистика:
- Q1 (25-й перцентиль): {q1:.1f}
- Q3 (75-й перцентиль): {q3:.1f}
- IQR (Interquartile Range): {iqr:.1f}
- Медиана: {median:.1f}
- Нижний ус: {lower_whisker:.1f}
- Верхний ус: {upper_whisker:.1f}
- Обнаружено выбросов: {len(outliers)}'''

ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", fc="lightgray", ec="gray", alpha=0.8))

plt.tight_layout()
plt.show()

# Дополнительная информация в консоли
print("Статистика boxplot:")
print(f"Q1 (25-й перцентиль): {q1:.2f}")
print(f"Q3 (75-й перцентиль): {q3:.2f}")
print(f"IQR: {iqr:.2f}")
print(f"Медиана: {median:.2f}")
print(f"Нижний ус (Q1 - 1.5×IQR): {lower_whisker:.2f}")
print(f"Верхний ус (Q3 + 1.5×IQR): {upper_whisker:.2f}")
print(f"Количество выбросов: {len(outliers)}")
```

<!-- #region editable=true slideshow={"slide_type": "slide"} -->
### Инструменты для работы: `feature-engine`

Библиотека `feature-engine` предоставляет удобные инструменты для обработки выбросов.

**Пример кода:**
```python
from feature_engine.outliers import Winsorizer

# Инициализация метода IQR
winsorizer = Winsorizer(capping_method='iqr', tail='both', fold=1.5)

# Обучение и преобразование данных
X_train_clean = winsorizer.fit_transform(X_train)

# Выбросы будут заменены на граничные значения
```
*   **`capping_method`**: `'iqr'` или `'gaussian'` (правило 3-х сигм).
*   **`tail`**: на каких «хвостах» искать (`'left'`, `'right'`, `'both'`).
*   **`fold`**: коэффициент (для IRR обычно 1.5).
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "slide"} -->
### Рекомендации

1.  **Анализ выбросов — обязательный этап** предобработки данных.
2.  **Всегда исследуйте природу выбросов** перед удалением: это ошибка или ценная аномалия?
3.  **Начинайте с простых методов** (IQR, Z-score) и визуализации.
4.  Для сложных случаев переходите к **многомерным методам**.
5.  Используйте специализированные библиотеки (например, `feature-engine`, `scikit-learn`), чтобы автоматизировать процесс.
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "slide"} jp-MarkdownHeadingCollapsed=true -->
## Заполнение пропусков в данных (Missing Data Imputation)

**Пропущенные значения (Missing Values)** — отсутствие данных для некоторых признаков у отдельных объектов.

**Причины возникновения:**
- Человек не заполнил поле в анкете
- Неразборчиво заполненные данные
- Сбои оборудования при сборе данных
- Технические ошибки при передаче данных

**Проблема:** Большинство моделей машинного обучения требуют полный вектор признаков без пропусков.

<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "slide"} jp-MarkdownHeadingCollapsed=true -->
### Базовые стратегии обработки

**1. Удаление объектов**
- Удалить все строки с пропущенными значениями
- **Подходит:** когда пропусков мало (<5% данных)
- **Не подходит:** когда пропусков много → потеря информации

**2. Заполнение значений (Imputation)**
- Заменить пропуски осмысленными значениями
- **Обязательно:** когда пропусков много
- **Требует:** выбора правильного метода заполнения
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "slide"} jp-MarkdownHeadingCollapsed=true -->
### Методы для категориальных признаков

**Категориальные признаки:** марка машины, профессия, город и т.д.

**1. Заполнение модой**
```python
# Самая частая категория
df['category'].fillna(df['category'].mode()[0])
```
- **Плюсы:** Простота
- **Минусы:** Может исказить распределение

**2. Новая категория "[пропуск]"**
```python
df['category'].fillna('[пропуск]')
```
- **Плюсы:** Сохраняет информацию о пропуске
- **Минусы:** Увеличивает размерность
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "slide"} jp-MarkdownHeadingCollapsed=true -->
### Продвинутые методы для категориальных признаков

**3. Предсказание классификатором**
```python
from sklearn.ensemble import RandomForestClassifier

# Обучаем модель предсказывать пропущенные значения
model = RandomForestClassifier()
model.fit(X_known, y_known)
predictions = model.predict(X_missing)
```

**Преимущества:**
- Учитывает взаимосвязи между признаками
- Более точное заполнение
- Адаптивное поведение

**Пример:** Предсказание города по месту работы
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "slide"} jp-MarkdownHeadingCollapsed=true -->
### Методы для вещественных признаков

**Вещественные признаки:** зарплата, возраст, температура и т.д.

**1. Заполнение средним значением**
```python
df['numeric'].fillna(df['numeric'].mean())
```
- **Чувствительно** к выбросам

**2. Заполнение медианой**
```python
df['numeric'].fillna(df['numeric'].median())
```
- **Устойчиво** к выбросам
- **Рекомендуется** по умолчанию
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "slide"} jp-MarkdownHeadingCollapsed=true -->
### Медиана более устойчива к выбросам!

<!-- #endregion -->

```python editable=true slideshow={"slide_type": "fragment"}
import numpy as np

# Нормальные данные
data = np.array([10, 12, 14, 15, 16, 18, 20])
print(f"Среднее: {data.mean():.1f}, Медиана: {np.median(data)}")

# Добавляем выброс
data_with_outlier = np.append(data, 100)
print(f"Среднее: {data_with_outlier.mean():.1f}, Медиана: {np.median(data_with_outlier)}")
```

<!-- #region editable=true slideshow={"slide_type": "slide"} jp-MarkdownHeadingCollapsed=true -->
### Условное заполнение и продвинутые методы

**Условное среднее/медиана**
```python
# Заполнение медианой по группам
df['salary'] = df.groupby('profession')['salary'].transform(
    lambda x: x.fillna(x.median()))
```

**Предсказание регрессией**
```python
from sklearn.linear_model import LinearRegression

# Предсказание пропущенного значения по другим признакам
model = LinearRegression()
model.fit(X_known, y_known)
predictions = model.predict(X_missing)
```
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "slide"} jp-MarkdownHeadingCollapsed=true -->
### Индикаторы пропусков

**Создание дополнительного признака**
```python
# Создаем индикатор пропуска
df['age_missing'] = df['age'].isna().astype(int)

# Заполняем пропуски
df['age'] = df['age'].fillna(df['age'].median())
```

**Преимущества:**
- Модель узнает о факте пропуска
- Может по-разному обрабатывать настоящие и заполненные значения
- Улучшает качество модели
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "slide"} -->
### Инструменты и библиотеки

**Pandas** - базовые операции
```python
df.fillna()          # Простое заполнение
df.interpolate()     # Интерполяция
```

**Scikit-learn** - продвинутые методы
```python
from sklearn.impute import SimpleImputer, KNNImputer

SimpleImputer(strategy='mean')    # Простое заполнение
KNNImputer()                      # K-ближайших соседей
```

**Feature-engine** - специализированные методы
```python
from feature_engine.imputation import ArbitraryNumberImputer
```
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "slide"} -->
### Рекомендации и лучшие практики

1. **Всегда анализируйте** природу пропусков перед заполнением
2. **Создавайте индикаторы** пропущенных значений
3. **Используйте медиану** вместо среднего для вещественных признаков
4. **Для категориальных данных** создавайте отдельную категорию
5. **Тестируйте разные методы** на ваших данных
6. **Документируйте** процесс обработки пропусков
<!-- #endregion -->
