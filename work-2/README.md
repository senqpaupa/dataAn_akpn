# Work 2 — API-пайплайн: отзывы → LLM → JSON

## Описание задачи

Скрипт читает отзывы клиентов из CSV-файла, отправляет их в LLM через HTTP API и сохраняет структурированный JSON-результат. По умолчанию используется OpenAI-compatible Chat Completions API: можно подключить OpenAI, OpenRouter, локальный прокси или другой сервис с совместимым endpoint. Для каждого отзыва модель определяет:

- тональность: `positive`, `negative` или `neutral`;
- основную тему отзыва;
- короткое объяснение;
- уверенность классификации от 0 до 1.

## Структура папки

```text
work-2/
  data/
    reviews.csv
  results/
    reviews_classified.json
  src/
    llm_review_pipeline.py
  .env_example
  .env        # локальный файл с секретами, не коммитится
  README.md
```

## Требования

- Python 3.10+
- API-ключ LLM-провайдера в локальном файле `work-2/.env`

Сторонние Python-библиотеки не нужны: скрипт использует только стандартную библиотеку.

## Настройка `.env`

Файл `work-2/.env` хранит чувствительные данные и добавлен в `.gitignore`, поэтому не должен попадать в репозиторий. В репозитории лежит только безопасный шаблон `work-2/.env_example`.

Для настройки скопируйте пример:

```bash
cp work-2/.env_example work-2/.env
```

Затем откройте `work-2/.env` и укажите настройки своего LLM-провайдера:

```env
LLM_API_KEY=ваш_api_ключ
LLM_BASE_URL=https://api.openai.com/v1/chat/completions
LLM_MODEL=gpt-4o-mini
LLM_PROVIDER_NAME=openai-compatible
LLM_RESPONSE_FORMAT=json_schema
```

Для другого OpenAI-compatible сервиса достаточно поменять `LLM_BASE_URL`, `LLM_MODEL` и ключ. Например, для шлюза OpenRouter endpoint будет отличаться, а модель может выглядеть как `openai/gpt-4o-mini` или другая модель, доступная в аккаунте.

Если провайдер не поддерживает строгий `json_schema`, попробуйте:

```env
LLM_RESPONSE_FORMAT=json_object
```

или самый совместимый вариант:

```env
LLM_RESPONSE_FORMAT=prompt_only
```

## Запуск

Из корня репозитория:

```bash
python3 work-2/src/llm_review_pipeline.py \
  --input work-2/data/reviews.csv \
  --output work-2/results/reviews_classified.json \
  --provider llm-api
```

Скрипт автоматически читает `work-2/.env`. Если нужно указать другой путь к env-файлу:

```bash
python3 work-2/src/llm_review_pipeline.py \
  --input work-2/data/reviews.csv \
  --output work-2/results/reviews_classified.json \
  --provider llm-api \
  --env-file work-2/.env
```

Параметры из `.env` можно переопределить аргументами командной строки:

```bash
python3 work-2/src/llm_review_pipeline.py \
  --input work-2/data/reviews.csv \
  --output work-2/results/reviews_classified.json \
  --provider llm-api \
  --api-url "https://api.openai.com/v1/chat/completions" \
  --model "gpt-4o-mini" \
  --response-format json_schema
```

В этой рабочей среде API-ключа OpenAI нет, поэтому приложенный файл результата был получен тем же скриптом через доступный LLM-интерфейс `codex-cli`:

```bash
python3 work-2/src/llm_review_pipeline.py \
  --input work-2/data/reviews.csv \
  --output work-2/results/reviews_classified.json \
  --provider codex-cli \
  --model gpt-5.5
```

## Пример входных данных

Файл [data/reviews.csv](data/reviews.csv):

```csv
id,review
1,"Курьер приехал вовремя, упаковка целая, товар полностью соответствует описанию."
2,"Приложение постоянно зависает при оплате, пришлось три раза перезапускать заказ."
```

## Пример выходных данных

Файл [results/reviews_classified.json](results/reviews_classified.json) содержит метаданные запуска и массив результатов:

```json
{
  "metadata": {
    "task": "review_sentiment_and_topic_classification",
    "llm_provider": "openai-compatible",
    "client": "llm-api",
    "api_url": "https://api.openai.com/v1/chat/completions",
    "model": "gpt-4o-mini",
    "input_file": "work-2/data/reviews.csv",
    "item_count": 10
  },
  "results": [
    {
      "id": "1",
      "sentiment": "positive",
      "topic": "delivery",
      "summary": "Клиент доволен своевременной доставкой и состоянием упаковки.",
      "confidence": 0.96
    }
  ]
}
```

## Описание полей результата

В корне JSON-файла есть два основных блока:

- `metadata` — техническая информация о запуске пайплайна.
- `results` — массив классифицированных отзывов.

Поля внутри `metadata`:

- `task` — название выполняемой задачи.
- `llm_provider` — тип или название LLM-провайдера.
- `client` — клиент, через который был выполнен запрос: например `llm-api` или `codex-cli`.
- `api_url` — endpoint API, если использовался HTTP API; для локального демонстрационного запуска может быть `null`.
- `model` — модель, которая обработала отзывы.
- `input_file` — путь к CSV-файлу с входными данными.
- `generated_at` — дата и время создания результата.
- `item_count` — количество обработанных отзывов.

Поля внутри каждого объекта в `results`:

- `id` — идентификатор отзыва из входного CSV.
- `sentiment` — тональность отзыва: `positive`, `negative` или `neutral`.
- `topic` — основная тема отзыва, например доставка, оплата, качество товара или поддержка.
- `summary` — короткое объяснение классификации на русском языке.
- `confidence` — оценка уверенности LLM от 0 до 1. Это не математически рассчитанная точность скрипта, а самооценка модели: чем ближе значение к 1, тем увереннее модель в своей классификации.

## Что делает скрипт

1. Читает CSV и проверяет наличие колонок `id` и `review`.
2. Формирует системный и пользовательский промпт для классификации отзывов.
3. Отправляет запрос в LLM Chat Completions API с требованием вернуть JSON по схеме.
4. Парсит JSON-ответ модели.
5. Проверяет, что в ответе есть результат для каждого входного отзыва.
6. Сохраняет итоговый JSON в файл.

## Ограничения

Скрипт не поддерживает вообще любой API-формат в мире: он рассчитан на OpenAI-compatible Chat Completions API с Bearer-токеном. Если у провайдера полностью другой протокол, например нестандартный endpoint или другой формат сообщений, понадобится небольшой адаптер. Качество классификации зависит от модели и формулировки отзывов. Для промышленного применения нужно добавить тестовую выборку с ручной разметкой и проверять точность модели на новых данных.
