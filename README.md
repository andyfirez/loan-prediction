# loan-prediction

Этот обучающий прект наглядно демонстрирует все этапы решения классической DS задачи классификации. В качестве примера взята задача кредитного скоринга, датасет взят отсюда: https://www.kaggle.com/datasets/ninzaami/loan-predication

Проект включает в себя следующие этапы:
1. Проверка на корректность и качество данных (файл baseline.ipynb)
2. Детальное изучение признаков и таргета (файл baseline.ipynb)
3. Изучение взаимосвязей между признаками (файл baseline.ipynb)
4. Препроцессинг данных (файл baseline.ipynb):
    - Обработка пропущенных значений
    - Кодирование категориальных переменных
    - Приведение данных к единому типу
5. Построение базовой модели и подсчет метрик (файл baseline.ipynb)
6. Преобразования данных, тестирование с помощью кросс-валидации. Выбор лучших преобразований для улучшения предсказаний (файл improvements_cv.ipynb)
7. Подбор гиперпараметров разными методами (файл hyperparam_tuning.ipynb)