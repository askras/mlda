---
jupyter:
  jupytext:
    split_at_heading: true
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

```python editable=true slideshow={"slide_type": ""}
from jupyterquiz import display_quiz

display_quiz(
    "quiz_10.json", #"questions_01.json",
    num=10,
    shuffle_questions=True,
    shuffle_answers=True,
    #preserve_responses=True,
    max_width=2000,
    #colors='fdsp',
    border_radius=False
)
```


