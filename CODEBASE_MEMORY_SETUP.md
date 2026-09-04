# Установка codebase-memory-mcp для Codex на macOS

Инструкция проверена 2026-09-04 по:

- https://github.com/DeusData/codebase-memory-mcp
- https://learn.chatgpt.com/docs/extend/mcp?surface=cli

## 1. Подготовка

Рабочий каталог пользователя в этом проекте — `code/`. Сначала сохранить
текущую конфигурацию Codex:

```bash
cp /Users/zuha/.codex/config.toml /Users/zuha/.codex/config.toml.before-codebase-memory
```

Проверить разрядность Mac:

```bash
uname -m
```

Для Apple Silicon ожидается `arm64`. Автоустановщик сам выбирает
нужную сборку.

## 2. Скачивание с проверкой скрипта

Не использовать непросмотренный `curl | bash`. Сначала скачать скрипт в
`Downloads`:

```bash
curl -fsSLo /Users/zuha/Downloads/codebase-memory-install.sh \
  https://raw.githubusercontent.com/DeusData/codebase-memory-mcp/main/install.sh
```

Просмотреть:

```bash
less /Users/zuha/Downloads/codebase-memory-install.sh
```

Выйти из `less`: `q`.

Запустить:

```bash
bash /Users/zuha/Downloads/codebase-memory-install.sh
```

Скрипт должен установить готовый бинарник, обнаружить Codex и добавить MCP-сервер
в `/Users/zuha/.codex/config.toml`. Ключ API не нужен.

## 3. Проверка установки

```bash
command -v codebase-memory-mcp
codex mcp list
```

В списке должен появиться `codebase-memory-mcp`. Если бинарник установлен, но MCP
не добавлен, выполнить:

```bash
codex mcp add codebase-memory-mcp -- "$(command -v codebase-memory-mcp)"
```

После этого повторить:

```bash
codex mcp list
```

Если команда `codex` в обычном терминале не найдена, это не означает, что
установка сломана: автоустановщик уже мог записать сервер в
`/Users/zuha/.codex/config.toml`. Проверить его после перезапуска можно командой
`/mcp` прямо в Codex. Для ручного добавления без CLI открыть настройки Codex,
раздел MCP servers, и добавить STDIO-сервер с командой, которую показывает:

```bash
command -v codebase-memory-mcp
```

## 4. Индексация проекта

В репозиторий уже добавлен `.cbmignore`: он исключает FITS, двоичные массивы,
готовые изображения, виртуальное окружение и сборочный мусор. Лёгкие CSV, JSON,
TXT, Markdown и TeX из `data/` и `runs/` намеренно оставлены для индексации,
если они не исключены основным `.gitignore`. Каталог `.codebase-memory/`
добавлен в `.gitignore`, чтобы сам граф не раздувал Git.

Это ограничение относится только к постоянному графу codebase-memory. Оно не
мешает Codex открывать FITS напрямую через Astropy, смотреть изображения или
читать любой файл из `runs/` при конкретной научной проверке.

Вариант A — после перезапуска Codex написать агенту:

> Проиндексируй текущий проект через codebase-memory-mcp, затем проверь статус индекса.

Вариант B — из `code/` вручную:

```bash
codebase-memory-mcp cli --progress index_repository \
  --repo-path "/Users/zuha/Desktop/FKI/4 курс 2025-2026/course_work-SBF"
```

Проверить индекс:

```bash
codebase-memory-mcp cli list_projects
```

Автоиндексацию при старте можно включить так:

```bash
codebase-memory-mcp config set auto_index true
```

Фоновое отслеживание изменений по умолчанию включено.

## 5. Сохранение контекста и диалога

`codebase-memory-mcp` индексирует репозиторий, но не читает историю этого чата
и не переносит скрытое состояние модели. Поэтому в проект уже добавлены два
человекочитаемых файла:

- `AGENTS.md` — постоянные правила работы;
- `PROJECT_MEMORY.md` — научные решения, параметры, результаты, ошибки и текущий TODO.

Их нужно закоммитить вместе с кодом. Для продолжения работы этого практичнее,
чем полный лог переписки: новый агент сначала получает компактный проверяемый
контекст, а затем находит определения и связи через индекс кода.

Для архивной копии исходной переписки использовать официальный экспорт:

1. Открыть профиль ChatGPT/Codex.
2. Выбрать `Settings` → `Data controls` → `Export data` → `Confirm export`.
3. Дождаться письма или SMS, скачать ZIP и проверить, что нужный диалог есть в
   истории. Подготовка архива может занять до семи дней, а ссылка действует 24 часа.
4. Хранить ZIP вне Git: экспорт может содержать не только этот проект и другие
   персональные данные.

Codex не даёт мне надёжного прямого доступа к сырому файлу всей текущей
переписки, поэтому автоматически выгрузить её один-в-один в репозиторий я не
могу. `PROJECT_MEMORY.md` — сделанный мной структурированный экспорт её
содержательной части.

## 6. Перезапуск Codex

1. Полностью закрыть Codex/ChatGPT desktop app через `Cmd+Q`.
2. Запустить приложение заново.
3. Открыть этот же диалог.
4. Выполнить `/mcp` и убедиться, что сервер виден и показывает 15 инструментов.

Новый чат не обязателен. Если в восстановленном диалоге `/mcp` не показывает
новый сервер, тогда создать новый чат в этом же проекте и написать:

> Прочитай `AGENTS.md` и `PROJECT_MEMORY.md`, проверь Git и статус F090W, затем продолжи с текущего TODO.

## 7. Откат

Удалить сервер из Codex:

```bash
codex mcp remove codebase-memory-mcp
```

Полностью удалить управляемую установку:

```bash
codebase-memory-mcp uninstall
```

При необходимости вернуть конфиг Codex из копии:

```bash
cp /Users/zuha/.codex/config.toml.before-codebase-memory /Users/zuha/.codex/config.toml
```
