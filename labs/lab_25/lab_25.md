---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.7
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Рекуррентные нейронные сети (RNN) и механизм внимания 

Продолжительность работы: - 4 часа.

Мягкий дедлайн (10 баллов): 11.05.2026

Жесткий дедлайн (5 баллов): 18.05.2026


**Цель работы**: 
- Изучить принципы работы рекуррентных сетей (RNN, LSTM, GRU)  
- Реализовать модель машинного перевода с механизмом внимания  
- Оценить качество генерации текста с помощью метрик (BLEU, Perplexity)  


## Заданипе 1. Подготовка данных для машинного перевода
Загрузите датасет `opus_books` (англо-русские предложения), проведите токенизацию и паддинг.  

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Загрузка данных (пример)
en_sentences = ["I love cats", "How are you?"]
ru_sentences = ["Я люблю кошек", "Как дела?"]

# Токенизация
tokenizer_en = Tokenizer()
tokenizer_en.fit_on_texts(en_sentences)
en_sequences = tokenizer_en.texts_to_sequences(en_sentences)

tokenizer_ru = Tokenizer()
tokenizer_ru.fit_on_texts(ru_sentences)
ru_sequences = tokenizer_ru.texts_to_sequences(ru_sentences)

# Паддинг
max_len = 10
en_padded = pad_sequences(en_sequences, maxlen=max_len, padding='post')
ru_padded = pad_sequences(ru_sequences, maxlen=max_len, padding='post')

# Здесь должен быть ваш код (реальная загрузка данных)

```

### Задание 2. Реализация LSTM-модели

Постройте модель Encoder-Decoder на LSTM для перевода текста.

```python
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.models import Model

# Параметры
vocab_size_en = len(tokenizer_en.word_index) + 1
vocab_size_ru = len(tokenizer_ru.word_index) + 1
embedding_dim = 256
latent_dim = 512

# Encoder
encoder_inputs = Input(shape=(None,))
enc_emb = Embedding(vocab_size_en, embedding_dim)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
_, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
dec_emb = Embedding(vocab_size_ru, embedding_dim)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(vocab_size_ru, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
```

```python
# Здесь должен быть ваш код (обучение модели)
```

### Задание 3. Добавление механизма внимания
Модифицируйте модель, добавив слой `Attention`.  

```python
from tensorflow.keras.layers import Attention, Concatenate

# Decoder с Attention
attention = Attention()([decoder_outputs, encoder_outputs])
decoder_concat = Concatenate()([decoder_outputs, attention])
decoder_outputs = decoder_dense(decoder_concat)
```

```python
# Здесь должен быть ваш код (обучение модели с вниманием)
```

### Задание 4. Визуализация матрицы внимания
Постройте тепловую карту для примера перевода.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Получение весов внимания
attention_model = Model(
    inputs=[encoder_inputs, decoder_inputs],
    outputs=attention_weights
)
attention_weights = attention_model.predict([en_test_seq, ru_test_seq])

# Визуализация
plt.figure(figsize=(10, 6))
sns.heatmap(attention_weights[0], cmap="YlGnBu")
plt.xlabel("Input tokens")
plt.ylabel("Output tokens")
plt.title("Attention Matrix")
plt.show()
```

```python
# Здесь должен быть ваш код (анализ конкретного примера)
```

### Задание 5. Оценка качества перевода

Вычислите BLEU-метрику для примера перевода.
(https://en.wikipedia.org/wiki/BLEU)

```python
from nltk.translate.bleu_score import sentence_bleu

# Пример оценки
reference = [["я", "люблю", "кошек"]]
candidate = ["я", "обожаю", "котов"]
bleu_score = sentence_bleu(reference, candidate)
print(f"BLEU score: {bleu_score:.2f}")
```

```python
# Здесь должен быть ваш код 
```

## **Контрольные вопросы**  
1. Чем LSTM отличается от обычной RNN?
2. Как работает механизм внимания в Seq2Seq?
3. Какие метрики используют для оценки генерации текста?
4. Почему матрица внимания часто диагональная?
5. В чем преимущество Teacher Forcing при обучении?
