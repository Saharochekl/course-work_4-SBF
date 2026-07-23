# Итоговый автоматический отчёт SBF

Сформирован: `2026-07-22T15:56:57+00:00`. Снимок первого прохода GO-7763: **завершён**.

## Жёсткий вывод

- GO-3055 даёт настоящую абсолютную калибровку: 14 измерений, привязанных к внешним модулям расстояния TRGB. Основная чистая выборка содержит 12 объектов.
- GO-7763 даёт плотную выборку одного скопления для проверки цветового наклона и систематики метода: готово 49 из 74, из них 40 новым однокольцевым выбором и 9 старым двухкольцевым методом.
- Единую регрессию GO-3055 + GO-7763 строить нельзя: цвета `F090W-F150W` и `F115W-F150W` не взаимозаменяемы. На графике они стоят рядом только как две отдельные задачи.
- Автоматика не заменяет просмотр изображений. Отдельно отмечены 25 неудачных обработок и 37 успешных, но подозрительных измерений; для 12 наиболее приоритетных успешных целей уже собраны пятикадровые диагностические листы.

## Состояние обработки

| Программа | Готово | Падение | В работе | Ожидает |
| --- | --- | --- | --- | --- |
| GO-3055 | 14 | 0 | 0 | 0 |
| GO-7763 | 49 | 25 | 0 | 0 |

У GO-7763 падений первого прохода: 25. Из них с явным `isolist too short`: 25. Это структурный сбой построения изофот, а не случайная сетевая ошибка; слепой повтор той же геометрии не считается исправлением.

## Модели

### GO-3055: абсолютная TRGB-калибровка

`y = -3.4542 +/- 0.0129 + (1.1654 +/- 0.2093) [color - 0.4000]; n=12, RMS=0.0610 mag (weighted_least_squares)`

Здесь `y = Mbar(F150W)`, цвет — `F090W-F150W`. Это модель, которую можно применять как калибровочную в пределах её выборки и фильтров, с сохранением оговорок по морфологии и диапазону цвета.

### GO-7763: Virgo, новый однокольцевой метод

Все новые успешные измерения:

`y = 28.1217 +/- 0.1136 + (-2.7065 +/- 1.5897) [color - 0.1500]; n=40, RMS=0.5867 mag (ordinary_least_squares)`

Только внутренний QC=PASS:

`y = 28.1143 +/- 0.1070 + (-2.7310 +/- 1.5246) [color - 0.1500]; n=38, RMS=0.5498 mag (ordinary_least_squares)`

Здесь `y = mbar(F150W)`, цвет — `F115W-F150W`. Это **не независимая калибровка расстояния**: модель проверяет наклон внутри Virgo и одновременно содержит глубину скопления, морфологию и систематику фиксированных колец.

## Кольца и систематика

Для нового GO-7763 медиана `inner - outer` = +0.1301 mag; объектов с `|inner - outer| > 0.2 mag`: 14 из 38. Если эта доля велика, итог нельзя лечить выбором одного удобного кольца: нужны кольца, масштабированные по размеру/профилю галактики.

## Что смотреть глазами

| Приоритет | Программа | Галактика | Статус | Причины |
| --- | --- | --- | --- | --- |
| 100 | GO-7763 | IC 3349 | failed | processing_failed; isophote_family_too_short |
| 100 | GO-7763 | IC 3363 | failed | processing_failed; isophote_family_too_short |
| 100 | GO-7763 | IC 3383 | failed | processing_failed; isophote_family_too_short |
| 100 | GO-7763 | IC 3388 | failed | processing_failed; isophote_family_too_short |
| 100 | GO-7763 | IC 3442 | failed | processing_failed; isophote_family_too_short |
| 100 | GO-7763 | IC 3466 | failed | processing_failed; isophote_family_too_short |
| 100 | GO-7763 | IC 3475 | failed | processing_failed; isophote_family_too_short |
| 100 | GO-7763 | IC 3492 | failed | processing_failed; isophote_family_too_short |
| 100 | GO-7763 | IC 3635 | failed | processing_failed; isophote_family_too_short |
| 100 | GO-7763 | NGC 4294 | failed | processing_failed; isophote_family_too_short |
| 100 | GO-7763 | NGC 4299 | failed | processing_failed; isophote_family_too_short |
| 100 | GO-7763 | NGC 4313 | failed | processing_failed; isophote_family_too_short |
| 100 | GO-7763 | NGC 4351 | failed | processing_failed; isophote_family_too_short |
| 100 | GO-7763 | NGC 4387 | failed | processing_failed; isophote_family_too_short |
| 100 | GO-7763 | NGC 4388 | failed | processing_failed; isophote_family_too_short |

Полная очередь лежит в [`manual_review_queue.csv`](manual_review_queue.csv), диагностические листы — в [`review_sheets/`](review_sheets/).

## Графики и таблицы

- [`01_campaign_status.png`](plots/01_campaign_status.png) — состояние обеих программ.
- [`02_color_models.png`](plots/02_color_models.png) — две отдельные цветовые модели.
- [`03_inner_vs_outer.png`](plots/03_inner_vs_outer.png) — согласие колец.
- [`04_annulus_delta_vs_color.png`](plots/04_annulus_delta_vs_color.png) — систематика колец против цвета.
- [`05_model_residuals.png`](plots/05_model_residuals.png) — остатки моделей и выбросы.
- [`06_qc_diagnostics.png`](plots/06_qc_diagnostics.png) — Pr/P0, формальная ошибка, стабильность k и расхождение колец.
- [`all_results.csv`](all_results.csv) — единая машиночитаемая таблица, но с явным разделением цветов и методов.
- [`model_summary.csv`](model_summary.csv) — коэффициенты, ошибки, RMS и ранговые тесты.
- [`analysis_summary.json`](analysis_summary.json) — полный снимок для воспроизводимости.
- [`campaign_attempts.csv`](campaign_attempts.csv), [`campaign_events.csv`](campaign_events.csv), [`campaign_artifacts.csv`](campaign_artifacts.csv) — история обработки из SQLite.
- [`campaign_resource_summary.csv`](campaign_resource_summary.csv) — максимальная память/своп и минимальный свободный диск по целям.

## Ограничения, которые нельзя замазывать автоматикой

1. Старые двухкольцевые и новые однокольцевые результаты GO-7763 методически неоднородны; старые девять не входят в основную Virgo-регрессию.
2. Фиксированные угловые кольца дают размер-зависимую систематику. До адаптивной геометрии это предварительная линейка, не финальная шкала расстояний.
3. Успешный численный результат не доказывает хорошую маску, центр и изофоты. Именно поэтому сохранена очередь ручной проверки и пятикадровые листы.
4. GO-7763 сам по себе не проверяет абсолютное расстояние без принятого общего модуля Virgo или независимых расстояний для отдельных объектов.
