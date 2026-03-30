---
jupyter:
  jupytext:
    split_at_heading: true
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python editable=true slideshow={"slide_type": ""}
from jupyterquiz import display_quiz

display_quiz(
    "exam.json", #"questions_01.json",
    #'exam.json',
    num=30,
    shuffle_questions=True,
    shuffle_answers=True,
    #preserve_responses=True,
    max_width=2000,
    #colors='fdsp',
    border_radius=False
)
```
```python
import json

files=[
    #'quiz_01.json',
    #'quiz_02.json',
    #'quiz_03.json',
    'quiz_04.json',
    'quiz_05.json',
    'quiz_06.json',
    'quiz_07.json',
    'quiz_08.json',
    'quiz_09.json',
    'quiz_10.json',
    'quiz_11.json',
]

def merge_JsonFiles(filename):
    result = list()
    for f1 in filename:
        with open(f1, 'r') as infile:
            result.extend(json.load(infile))

    with open('exam.json', 'w') as output_file:
        json.dump(result, output_file)

merge_JsonFiles(files)

```






