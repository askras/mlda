---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region id="ZdIOldj_aag8" -->
# Основные виды нейросетей (CNN и RNN)
<!-- #endregion -->

<!-- #region id="PLYbee6p9xXZ" -->
### Продолжительность и сроки сдачи

Продолжительность работы: - 4 часа.

Мягкий дедлайн (5 баллов): 11.04.2024

Жесткий дедлайн (2.5 баллов): 25.04.2024
<!-- #endregion -->

<!-- #region id="PLYbee6p9xXZ" -->
**Цель работы:** реализация сверточных и рекуррентных сетей, исследование проблемы затухающих и взрывающихся градиентов.
<!-- #endregion -->

<!-- #region id="rAUuvVjg9xXc" -->
## Сверточные сети

Вернемся в очередной раз к датасету MNIST. Для начала загрузим данные и определим несколько полезных функций как на прошлом семинаре.
<!-- #endregion -->

```python id="diIxeLEK9xXe"
# Загрузка зависимостей
import util

%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import random
from IPython.display import clear_output

import torch
import torch.nn as nn
import torch.nn.functional as F
```

```python id="diIxeLEK9xXe"
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
```

```python id="epP3TUcA9xXk"
from util import load_mnist
X_train, y_train, X_val, y_val, X_test, y_test = load_mnist(flatten=True)

plt.figure(figsize=[6, 6])
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.title(f"Label: {y_train[i]}")
    plt.imshow(X_train[i].reshape([28, 28]), cmap='gray');
```

```python id="4o8KJ0Rh9xXq"
from util import iterate_minibatches

def train_epoch(model, optimizer, batchsize=32):
    loss_log, acc_log = [], []
    model.train()
    for x_batch, y_batch in iterate_minibatches(X_train, y_train, batchsize=batchsize, shuffle=True):
        data = torch.from_numpy(x_batch.astype(np.float32))
        target = torch.from_numpy(y_batch.astype(np.int64))

        optimizer.zero_grad()
        output = model(data)
        
        pred = torch.max(output, 1)[1].numpy()
        acc = np.mean(pred == y_batch)
        acc_log.append(acc)
        
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        loss = loss.item()
        loss_log.append(loss)
    return loss_log, acc_log

def test(model):
    loss_log, acc_log = [], []
    model.eval()
    for x_batch, y_batch in iterate_minibatches(X_val, y_val, batchsize=32, shuffle=True):
        data = torch.from_numpy(x_batch.astype(np.float32))
        target = torch.from_numpy(y_batch.astype(np.int64))

        output = model(data)
        loss = F.nll_loss(output, target)
        
        pred = torch.max(output, 1)[1].numpy()
        acc = np.mean(pred == y_batch)
        acc_log.append(acc)
        
        loss = loss.item()
        loss_log.append(loss)
    return loss_log, acc_log

def plot_history(train_history, val_history, title='loss'):
    plt.figure()
    plt.title(title)
    plt.plot(train_history, label='train', zorder=1)
    
    points = np.array(val_history)
    
    plt.scatter(points[:, 0], points[:, 1], marker='+', s=180, c='orange', label='val', zorder=2)
    plt.xlabel('train steps')
    
    plt.legend(loc='best')
    plt.grid()

    plt.show()
    
def train(model, opt, n_epochs):
    train_log, train_acc_log = [], []
    val_log, val_acc_log = [], []

    batchsize = 32

    for epoch in range(n_epochs):
        print(f"Epoch {epoch} of {n_epochs}")
        train_loss, train_acc = train_epoch(model, opt, batchsize=batchsize)

        val_loss, val_acc = test(model)

        train_log.extend(train_loss)
        train_acc_log.extend(train_acc)

        steps = len(X_train) / batchsize
        val_log.append((steps * (epoch + 1), np.mean(val_loss)))
        val_acc_log.append((steps * (epoch + 1), np.mean(val_acc)))

        clear_output()
        plot_history(train_log, val_log)    
        plot_history(train_acc_log, val_acc_log, title='accuracy')
        print(f"Epoch {epoch} error = {1 - val_acc_log[-1][1]:.2%}")
            
    print("Final error: {:.2%}".format(1 - val_acc_log[-1][1]))
```

<!-- #region id="E9gyKDDd9xXw" -->
**Задание 1:** 
Реализуйте сверточную сеть, которая состоит из двух последовательных применений свертки, relu и max-пулинга, а потом полносвязного слоя. 

Подберите параметры так, чтобы на выходе последнего слоя размерность тензора была 4 x 4 x 16. 
В коде ниже используется обертка nn.Sequential, ознакомьтесь с ее интерфейсом.

Добейтесь, чтобы ошибка классификации после обучения (см. ниже) была не выше 1.5%.
<!-- #endregion -->

```python id="GcpwpeTE9xXy"
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.features = nn.Sequential(
            # <your code here>
        )
        
        self.classifier = nn.Linear(4 * 4 * 16, 10)
        
    def forward(self, x):
        # <your code here>
        return F.log_softmax(out, dim=-1)
```

<!-- #region id="6jDp1wX99xX3" -->
Посчитаем количество обучаемых параметров сети.
<!-- #endregion -->

```python id="Rwu1hOND9xX6"
def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])

model = ConvNet()
print("Total number of trainable parameters:", count_parameters(model))
```

```python id="-65n280K9xX_"
%%time

opt = torch.optim.RMSprop(model.parameters(), lr=0.001)
train(model, opt, 5)
```

<!-- #region id="PJFnA-sN9xYK" -->
Мы с легкостью получили качество классификаци лучше, чем было раньше. 
На самом деле для более честного сравнения нужно поисследовать обе архитектуры и подождать побольше итераций до сходимости, но в силу ограниченности вычислительных ресурсов это может не получиться. 
Результаты из которых "выжали максимум" можно посмотреть например, на этой странице: http://yann.lecun.com/exdb/mnist/, и там видно, что качество сверточных сетей гораздо выше. 
А если работать с более сложными изоражениями (например, с ImageNet), то сверточные сети побеждают с большим отрывом.
<!-- #endregion -->

<!-- #region id="hP6KOZ729xYP" -->
## Рекуррентные сети

Для рекуррентных сетей используем датасет с именами и будем определять из какого языка произошло данное имя. Для этого построим рекуррентную сеть, которая с именами на уровне символов. Для начала скачаем файлы и конвертируем их к удобному формату (можно не особо вникать в этот код).
<!-- #endregion -->

```python id="n5wPAmAy9xYR"
# На Windows придется скачать архив по ссылке (~3Mb) и распаковать самостоятельно
!wget --quiet --show-progress https://download.pytorch.org/tutorial/data.zip
!unzip -q -f ./data.zip
```

```python id="ZKxrnmgW9xYY"
from io import open
import glob

def findFiles(path): return glob.glob(path)

print(findFiles('data/names/*.txt'))

import unicodedata
import string

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

print(unicodeToAscii('Ślusàrski'))

# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('data/names/*.txt'):
    category = filename.split('/')[-1].split('.')[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0][0]
    return all_categories[category_i], category_i
```

<!-- #region id="319MdTEC9xYe" -->
Определим несколько удобных функций для конвертации букв и слов в тензоры.
<!-- #endregion -->

<!-- #region id="319MdTEC9xYe" -->
**Задание 2**:
напишите последнюю функцию для конвертации слова в тензор.
<!-- #endregion -->

```python id="R3fUjiX_9xYg"
# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    # <your code here>

print(letterToTensor('J'))
print(lineToTensor('Jones').size())
```

<!-- #region id="4bcXqGXx9xYl" -->
**Задание 3:** 
Реализуйте однослойную рекуррентную сеть.
<!-- #endregion -->

```python id="whTbgC5R9xYn"
class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNCell, self).__init__()
        
        self.hidden_size = hidden_size
        # <your code here>

    def forward(self, input, hidden):
        # <your code here>
        return hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 128
rnncell = RNNCell(n_letters, n_hidden)
```

<!-- #region id="6rnAXB009xYr" -->
Предсказание будем осуществлять при помощи линейного класссификатора поверх скрытых состояний сети.
<!-- #endregion -->

```python id="j6rWQ4_a9xYt"
classifier = nn.Sequential(nn.Linear(n_hidden, n_categories), nn.LogSoftmax(dim=1))
```

<!-- #region id="B7KVpW449xYz" -->
Проверим, что все корректно работает: выходы классификаторы должны быть лог-вероятностями.
<!-- #endregion -->

```python id="r9fgCAXd9xY1"
input = letterToTensor('A')
hidden = torch.zeros(1, n_hidden)

output = classifier(rnncell(input, hidden))
print(output)
print(torch.exp(output).sum())
```

```python id="ATHA1a8J9xY6"
input = lineToTensor('Albert')
hidden = torch.zeros(1, n_hidden)

output = classifier(rnncell(input[0], hidden))
print(output)
print(torch.exp(output).sum())
```

<!-- #region id="HV0I9pd99xY-" -->
Для простоты в этот раз будем оптимизировать не по мини-батчам, а по отдельным примерам. Ниже несколько полезных функций для этого.
<!-- #endregion -->

```python id="97WUqv-O9xZB"
import random

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('category =', category, '/ line =', line)
```

<!-- #region id="o8KAO_JP9xZG" -->
**Задание 4:** 
Реализуйте вычисление ответа в функции train. Если все сделано правильно, то точность на обучающей выборке должна быть не менее 70%.
<!-- #endregion -->

```python id="BzuMsHyD9xZG"
from tqdm.auto import tqdm

def train(category, category_tensor, line_tensor, optimizer):
    hidden = rnncell.initHidden()

    rnncell.zero_grad()
    classifier.zero_grad()

    # <your code here> use rnncell and classifier

    loss = F.nll_loss(output, category_tensor)
    loss.backward()
    optimizer.step()
    
    acc = (categoryFromOutput(output)[0] == category)

    return loss.item(), acc

n_iters = 50000
plot_every = 1000

current_loss = 0
all_losses = []
current_acc = 0
all_accs = []

n_hidden = 128

rnncell = RNNCell(n_letters, n_hidden)
classifier = nn.Sequential(nn.Linear(n_hidden, n_categories), nn.LogSoftmax(dim=1))
params = list(rnncell.parameters()) + list(classifier.parameters())
opt = torch.optim.RMSprop(params, lr=0.001)
for iter in tqdm(range(1, n_iters + 1)):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    loss, acc = train(category, category_tensor, line_tensor, opt)
    current_loss += loss
    current_acc += acc

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0
        all_accs.append(current_acc / plot_every)
        current_acc = 0
        
plt.figure()
plt.title("Loss")
plt.plot(all_losses)
plt.grid()
plt.show()

plt.figure()
plt.title("Accuracy")
plt.plot(all_accs)
plt.grid()
plt.show()
```

<!-- #region id="81DCBYA69xZL" -->
## Затухающие и взрывающиеся градиенты

Эксперименты будем проводить опять на датасете MNIST, но будем работать с полносвязными сетями. В этом разделе мы не будем пытаться подобрать более удачную архитектуру, нам интересно только посмотреть на особенности обучения глубоких сетей.
<!-- #endregion -->

```python id="lj2o7YMV9xZL"
from util import load_mnist
X_train, y_train, X_val, y_val, X_test, y_test = load_mnist(flatten=True)
```

<!-- #region id="gj-hnp_B9xZP" -->
Для экспериментов нам понадобится реализовать сеть, в которой можно легко менять количество слоев. Также эта сеть должна сохранять градиенты на всех слоях, чтобы потом мы могли посмотреть на их величины.
<!-- #endregion -->

<!-- #region id="gj-hnp_B9xZP" -->
**Задание 5:** допишите недостающую часть кода ниже.
<!-- #endregion -->

```python id="D_zBqPek9xZQ"
class DeepDenseNet(nn.Module):
    def __init__(self, n_layers, hidden_size, activation):
        super().__init__()
        self.activation = activation
        
        l0 = nn.Linear(X_train.shape[1], hidden_size)
        self.weights = [l0.weight]
        self.layers = [l0]
        
        # <your code here>
        
        self.seq = nn.Sequential(*self.layers)
        
        for l in self.weights:
            l.retain_grad()
        
    def forward(self, x):
        out = self.seq(x)
        return F.log_softmax(out, dim=-1)
```

<!-- #region id="3vd1b00P9xZV" -->
Модифицируем наши функции обучения, чтобы они также рисовали графики изменения градиентов.
<!-- #endregion -->

```python id="k-4-vgt29xZY"
import scipy.sparse.linalg

def train_epoch_grad(model, optimizer, batchsize=32):
    loss_log, acc_log = [], []
    grads = [[] for l in model.weights]
    model.train()
    for x_batch, y_batch in iterate_minibatches(X_train, y_train, batchsize=batchsize, shuffle=True):
        # data preparation
        data = torch.from_numpy(x_batch.astype(np.float32))
        target = torch.from_numpy(y_batch.astype(np.int64))

        optimizer.zero_grad()
        output = model(data)
        
        pred = torch.max(output, 1)[1].numpy()
        acc = np.mean(pred == y_batch)
        acc_log.append(acc)
        
        loss = F.nll_loss(output, target)
        # compute gradients
        loss.backward()
        # make a step
        optimizer.step()
        loss = loss.item()
        loss_log.append(loss)
        
        for g, l in zip(grads, model.weights):
            g.append(np.linalg.norm(l.grad.numpy()))
    return loss_log, acc_log, grads


def train_grad(model, opt, n_epochs):
    train_log, train_acc_log = [], []
    val_log, val_acc_log = [], []
    grads_log = None

    batchsize = 32

    for epoch in range(n_epochs):
        print(f"Epoch {epoch} of {n_epochs}")
        train_loss, train_acc, grads = train_epoch_grad(model, opt, batchsize=batchsize)
        if grads_log is None:
            grads_log = grads
        else:
            for a, b in zip(grads_log, grads):
                a.extend(b)

        val_loss, val_acc = test(model)

        train_log.extend(train_loss)
        train_acc_log.extend(train_acc)

        steps = len(X_train) / batchsize
        val_log.append((steps * (epoch + 1), np.mean(val_loss)))
        val_acc_log.append((steps * (epoch + 1), np.mean(val_acc)))

        # display all metrics
        clear_output()
        plot_history(train_log, val_log)    
        plot_history(train_acc_log, val_acc_log, title='accuracy')    

        plt.figure()
        all_vals = []
        for i, g in enumerate(grads_log):
            w = np.ones(100)
            w /= w.sum()
            vals = np.convolve(w, g, mode='valid')
            plt.semilogy(vals, label=str(i+1), color=plt.cm.coolwarm((i / len(grads_log))))
            all_vals.extend(vals)
        plt.legend(loc='best')
        plt.grid()
        plt.show()
```

<!-- #region id="J5fobtIJ9xZc" -->
**Задание 6:**
* Обучите сети глубины 10 и больше с сигмоидой в качестве активации. Исследуйте, как глубина влияет на качество обучения и поведение градиентов на далеких от выхода слоях.
* Теперь замените активацию на ReLU и посмотрите, что получится.
<!-- #endregion -->

```python id="ASylWcPc9xZd"
# ...
```

<!-- #region id="PJFnA-sN9xYK" -->
**Контрольные вопросы:** 
* Почему сверточные сети обладают таким преимуществом именно для изображений?
* Почему несмотря на малое количество параметров обучение сверточных сетей занимает так много времени?
<!-- #endregion -->
