# Описание

Ноутбук `LAVKA_INCREMENTAL_LOADER.ipynb` построен на основе исходного `LAVKA_LOADER.ipynb`, но переведен с полной перезаписи (`overwrite`) на инкрементальную загрузку.

Цель:
- загружать только новые файлы;
- сохранять историю изменения метрик;
- поддерживать регулярную weekly-поставку данных с 10-дневным перекрытием.

Источники:
- PO1: `.../DOS REP/Метрики cpfr/`
- PO1 archive: `.../DOS REP/Метрики cpfr/archiv`
- PO1 KUB: `.../Для КУБ Лавка/`
- WBD: `metrics_ao`, `metrics_orders`, `metrics_osa`, `metrics_prediction`, `metrics_sales_stock`
- справочники: `DICTIONARIES`, `directory/products.csv`

Выход:
- загрузка в таблицы `ECOM_ETL.LAVKA_PO1` и `ECOM_ETL.LAVKA_WBD_*`;
- для справочников оставлен `overwrite`;
- staging-папки в DBFS очищаются по содержимому после завершения.
