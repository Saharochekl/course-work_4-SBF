Загрузка и служебный код / Downloads and infrastructure
======================================================

Русский
-------

Все команды выполняются из ``code/`` в активном окружении проекта.
Поддерживаются macOS/Linux. ``psutil`` обязателен для контроля ресурсов,
``Astropy`` — для проверки FITS. Научные notebook этими командами не запускаются.

1. ``py download_go3055_go7763.py --program 3055`` — проверить локальные входы.
   Добавление ``--download`` явно разрешает сеть. Загружаются обе полосы из
   принятого manifest, поэтому отдельный загрузчик изображений F090W не нужен.
2. ``py download_wss_opds.py --program 3055`` — проверить локальные WSS OPD.
   ``--download`` разрешает STPSF/MAST получить недостающие OPD. Без этого флага
   даже импорт STPSF отложен, чтобы его проверка reference data не обращалась в сеть.
3. ``py -m unittest test_sbf_campaign test_sbf_target_status
   test_download_go3055_go7763 test_download_wss_opds`` — быстрые проверки.
   HTTP-тесты поднимают только временный сервер на ``127.0.0.1``.

Назначение модулей
~~~~~~~~~~~~~~~~~

* ``download_go3055_go7763``: manifest → проверка файлов → ограниченная очередь
  HTTP → проверка FITS → атомарное переименование. ``.part`` докачивается только
  при совпадении архивной идентичности; новый ответ никогда не дописывается
  вслепую к старой версии файла. Повтор запуска готовые файлы не скачивает.
* ``download_wss_opds``: группирует даты изображений и находит ближайший локальный
  OPD. JSON-отчёт хранит реальное временное расстояние до OPD; допуск не означает
  доказанную стабильность PSF на протяжении всего этого времени.
* ``sbf_campaign_runtime``: группы процессов, SIGINT/TERM, дедлайн, RAM/диск,
  проверка FITS и общий атомарный writer. Запись идёт через временный файл в том же
  каталоге, ``fsync`` и ``replace``: прерывание не оставляет половину итогового JSON.
* ``sbf_campaign_state``: транзакционный SQLite/WAL-журнал F150W. Разрешённые
  переходы состояний не дают назвать незаконченную работу успешной. После
  прерывания активные задания и попытки вместе получают ``INTERRUPTED``.
* ``sbf_target_status``: читаемый CSV и проверка текущего контракта SBF-2.
  Проверяются галактика, фильтры, пять необходимых FITS, наличие таблиц и конечная
  SBF-величина. F090W имеет собственный resume-контракт в своём обработчике.

Идентичность задания определяется наблюдением, фильтрами и архивными URI.
Хэши кода остаются provenance, а не причиной потерять уже выполненное задание.
Это не разрешение переиспользовать несовместимую обработку: обработчики отдельно
проверяют конфигурацию, исходные файлы и научные продукты.

Параметры и причины
~~~~~~~~~~~~~~~~~~

Это эксплуатационные ограничения, не дополнительные параметры SBF-фита.
Научные константы описаны отдельно в документации измерительного контура.

* FITS: блок **2880 байт**, карточка **80 байт** — определения формата.
  Проверка заголовка по сети ограничена **32 блоками** (92 160 байт), чтобы не
  скачивать гигабайтный кадр ради даты; отсутствие ``END`` — явная ошибка.
* ``LOCAL_MANIFEST_SIZE_TOLERANCE=0.01``: терпимость к небольшому изменению
  архивного заголовка, только после проверки FITS. Известный HTTP-размер обязан
  совпасть точно. Этот допуск не допускает усечённый файл.
* Загрузчик: **4** потока по умолчанию, максимум **16** — ограничение нагрузки;
  **1 MiB** на чтение — ограничение RAM/задержки отмены; **40 GiB** резерва — место
  для временных FITS и другой работы. Всё меняется через CLI.
* **8** попыток, тайм-аут **120 s**: ограниченное восстановление после обрыва.
  Повторы для 408/425/429/500/502/503/504; постоянные HTTP-ошибки не повторяются.
  Задержка **5–120 s** с разбросом **±10%** разводит параллельные повторы;
  серверный ``Retry-After`` имеет приоритет. Прогресс печатается каждые **10 s**.
* OPD: **30 дней** — настраиваемый предел покрытия локального каталога, не
  предположение о физической неизменности оптики; **30 s** — тайм-аут tiny header
  request. ``median-first`` — порядок, снижающий число необходимых OPD.
* Супервизор: выборка ресурсов раз в **30 s**, обычная нехватка RAM в **3**
  последовательных выборках — защита от кратковременного скачка. Аварийная RAM,
  предел RSS и диска останавливают сразу. Недоступное обязательное измерение
  тоже останавливает, а не подменяется нулём.
* Завершение: **30 s** после TERM для записи состояния, ещё **5 s** после KILL;
  проверка завершения раз в **0.1 s**, минимальное ожидание главного цикла
  **0.01 s** исключает активное вращение CPU.
* SHA256 читается блоками **1 MiB**, независимо от размера FITS. SQLite ждёт
  чужого writer до **30 s**. ``SCHEMA_VERSION=1`` — текущий формат БД, а не версия
  научного метода. Списки колонок, ключей FITS и состояний — контракты файлов.
* ``PROJECT_ROOT``/пути по умолчанию выводятся из местоположения кода.
  URL MAST и User-Agent идентифицируют сервис и клиент, не меняют измерения.

Не удалённые меры защиты
~~~~~~~~~~~~~~~~~~~~~~~

Оставлены докачка HTTP Range, проверка ETag/Last-Modified, карантин сомнительной
частичной загрузки, fallback на PID при опасной общей группе процессов и
ограниченные повторы сети. Это реальные режимы отказа, а не попытки скрыть ошибку.
Самодельные парсеры памяти ``/proc``/``sysctl``, Windows-ветви, metadata-only
проверка FITS, SBF-3-специфические проверки и неиспользуемые SQLite-обёртки убраны
из активного кода. Старые реализации и их тесты сохранены в локальном ``legacy``.

English
-------

Run commands from ``code/`` in the project environment; macOS/Linux are supported.
``psutil`` is required for resource monitoring and Astropy for FITS validation.

* ``download_go3055_go7763.py --program 3055`` inventories both science bands
  without network access. Add ``--download`` explicitly to fetch missing inputs.
  Safe HTTP Range resumes require matching remote identity; complete files skip
  network access. Publication follows size/header/checksum validation.
* ``download_wss_opds.py --program 3055`` checks local WSS coverage. STPSF is
  imported only with ``--download`` to keep its own reference checks out of dry
  runs. The report records actual OPD age, not just pass/fail.
* ``sbf_campaign_runtime`` owns process groups, resource limits, deadlines,
  FITS validation, hashing and atomic writes. Missing monitored resources stop
  a guarded worker rather than silently disabling a configured limit.
* ``sbf_campaign_state`` owns the F150W SQLite/WAL queue. Transactional recovery
  marks both jobs and attempts interrupted. Stable observation/filter/URI job
  identity is separate from provenance hashes; runners still check compatible
  input/configuration/product contracts before scientific reuse.
* ``sbf_target_status`` owns the readable CSV and current SBF-2 result contract.
  F090W resume validation belongs to its runner, not this CSV adapter.

Default rationale: FITS uses 2880-byte blocks/80-byte cards; the header probe is
32 blocks. A 1% local manifest-size tolerance covers small archive revisions only
after structural validation; authoritative HTTP lengths require exact matches.
Four download workers (maximum 16), 1-MiB reads and a 40-GiB reserve bound laptop
I/O, RAM and disk usage. Eight attempts, a 120-s socket timeout and 5–120-s retry
backoff with ±10% jitter handle transient HTTP failures; Retry-After takes
precedence. Progress is printed every 10 s. The OPD age cap is 30 days and the
header timeout is 30 s; neither is a fitted optical parameter.

Supervision samples every 30 s, tolerates three consecutive low-RAM samples,
and gives TERM/KILL 30/5 s respectively. Emergency limits act immediately;
0.1-s termination polling and 0.01-s minimum loop waits avoid busy spinning.
Hash reads use 1 MiB. SQLite waits at most 30 s for another writer. Schema version
1, CSV columns, FITS keys and allowed states define storage contracts. Root paths
derive from the checkout; MAST URL/User-Agent identify the service/client.

Keep the explicit network retry/resume, quarantine and safe PID-only termination
paths: each corresponds to a real failure condition. Obsolete memory parsers,
unsupported Windows branches, metadata-only FITS validation and SBF-3-only code
are no longer active. Archived tests are separate; the active test suite uses
synthetic files and a temporary localhost server, not historical science FITS.
