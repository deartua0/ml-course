# Нейросеть на NumPy

Базовые слои, функции потерь и оптимизаторы из задания `hw08_nn_from_scratch`.

- `modules.ipynb` — реализация в исходном ноутбуке.
- `modules.py` — те же базовые реализации в Python; файл для сдачи.
- `test_modules.ipynb` — исходные проверки со сравнением с PyTorch.
- `run_tests.py` — запуск базовых проверок против `modules.py`.
- `test_gradients.py` — численные производные, устойчивость и пробное обучение на синтетических данных; нужен только NumPy.
- `main_notebook.ipynb`, `mnist.py` — исходные материалы для последующих экспериментов с MNIST в лабораторной 2; эксперименты ещё не заполнены.

Реализованы Linear, Sequential, SoftMax, LogSoftMax, BatchNormalization,
ChannelwiseScaling, Dropout, ReLU, LeakyReLU, ELU, SoftPlus, MSE, оба варианта
NLL, SGD с momentum и Adam. Каждый слой сохраняет выход и градиент по входу;
параметры обновляются отдельно оптимизатором. BatchNormalization и Dropout
поддерживают режимы `train()` и `evaluate()`.

Необязательные Conv2d и MaxPool2d оставлены как исходные заготовки в ноутбуке;
в базовый `modules.py` они не входят.

## Проверка

Численные проверки:

```bash
python -m pip install numpy
python test_gradients.py
```

Исходные сравнения с PyTorch:

```bash
python -m pip install torch
python run_tests.py
```

Команды выполняются из этой папки. Для запуска исходного ноутбука с
дополнительными свёрточными слоями также требуются SciPy и scikit-image.

## Источник

[Условие и исходные материалы girafe-ai/ml-course, ветка 23f_basic](https://github.com/girafe-ai/ml-course/tree/23f_basic/homeworks/hw08_nn_from_scratch).
Авторские пояснения и атрибуция материалов сохранены в ноутбуках.
