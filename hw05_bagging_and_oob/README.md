* assignment_bagging_and_oob: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/girafe-ai/ml-course/blob/23f_basic/homeworks/hw05_bagging_and_oob/assignment_bagging_and_oob.ipynb)

__Контест__: https://contest.yandex.ru/contest/54645

## Реализация и проверка

`bagging.py` и ноутбук содержат одну реализацию: bootstrap-выборки длины n,
отдельная модель на каждой выборке, средний прогноз и OOB-MSE.
Для OOB сначала усредняются прогнозы моделей, не видевших объект, затем
считается квадрат ошибки. Объекты без OOB-прогнозов получают `None`
и исключаются из MSE; если таких прогнозов нет вообще, возникает `ValueError`.

В ноутбуке выполнены все исходные проверки: Simple (100 итераций),
Medium (10) и Complex (10). Для воспроизводимости задан `np.random.seed(42)`.
Зависимости ноутбука: `numpy`, `scikit-learn`, `tqdm`.

Дополнительные проверки небольших примеров и граничных случаев:

```bash
python -m unittest discover -s hw05_bagging_and_oob -p 'test_bagging.py' -v
```

Команда запускается из корня репозитория. Для этих проверок достаточно `numpy`.
