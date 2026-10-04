# Loan Approval Prediction

Production-oriented ML-проект для бинарного предсказания одобрения кредитной заявки.

## Результат baseline

Модель `LogisticRegression` с preprocessing pipeline показала validation ROC-AUC **0.90681**.

## Технологии

- Python 3.12
- Poetry
- pandas и scikit-learn
- pytest и pytest-cov
- Ruff
- mypy
- pre-commit

## Установка

```bash
git clone [https://github.com/ILKA777/Loan-Approval-Prediction.git](https://github.com/ILKA777/Loan-Approval-Prediction.git)
cd Loan-Approval-Prediction

poetry install
poetry run pre-commit install
```

## Данные

Для запуска в корне проекта должны находиться:

```text
train.csv
test.csv
sample_submission.csv
```

## Проверка качества кода

```bash
make check
```

Или отдельными командами:

```bash
poetry check
poetry run ruff format --check src tests
poetry run ruff check src tests
poetry run mypy src
poetry run pytest
```

## Обучение

```bash
make train
```

Модель будет сохранена в `models/loan_approval_model.joblib`.

## Предсказание

```bash
make predict
```

Результат будет сохранён в `reports/submission.csv`.

## Структура

```text
src/loan_approval/  # Production-код: загрузка данных, признаки, train, predict
tests/              # Unit-тесты
models/             # Локальные обученные модели, исключены из Git
reports/            # Локальные prediction/submission-файлы, исключены из Git
pyproject.toml      # Зависимости и конфигурации инструментов
poetry.lock         # Зафиксированные версии зависимостей
.pre-commit-config.yaml
```
