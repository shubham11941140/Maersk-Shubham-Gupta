# Data dictionary

The warehouse is the DuckDB file `data/warehouse/supply_chain.duckdb`, built by `python -m pipeline`. The README section *Data pipeline and data model* covers layers, keys and relationships. This file defines every column.

Conventions:

- All timestamps are naive UTC (assumed: the source carries no offsets).
- `dq_flags` lists the fixes and flags applied to a row during curation. An empty list means the row was clean.
- Columns starting with `_` are lineage, not business data.

## `curated.shipments`: one row per shipment

| Column | Type | Null? | Definition / rule |
|---|---|---|---|
| `shipment_id` | VARCHAR | no | **PK**. `SHP-#####`. Unique after removing exact duplicates and quarantining conflicting versions. |
| `customer_id` | VARCHAR | no | `CUST-####` |
| `origin_port` | VARCHAR | no | **FK → `curated.ports.port_code`**. Upper-cased and trimmed. |
| `destination_port` | VARCHAR | no | **FK → `curated.ports.port_code`** |
| `route_key` | VARCHAR | no | `origin_port || ' → ' || destination_port` |
| `vessel_id` | VARCHAR | no | `VSL-####`. Soft link to `curated.port_events.vessel_id`. |
| `booking_date` | TIMESTAMP | no | When the booking was made |
| `planned_departure` | TIMESTAMP | no | ≥ `booking_date` (contract rule) |
| `actual_departure` | TIMESTAMP | yes | NULL when not yet departed, cancelled, or discarded because it was before `booking_date` (flag `actual_departure_before_booking`) |
| `planned_arrival` | TIMESTAMP | no | > `planned_departure` |
| `actual_arrival` | TIMESTAMP | yes | NULL when there is no observed arrival |
| `container_count` | INTEGER | yes | 1–100. Zero and outliers above 100 are set to NULL and flagged. |
| `cargo_type` | VARCHAR | no | One of 10 canonical categories, or `Unknown` |
| `weight_tons` | DOUBLE | yes | > 0. Missing, zero and negative values are NULL and flagged. |
| `status` | VARCHAR | no | `DELIVERED` / `DELAYED` / `CANCELLED` / `UNKNOWN`, re-derived from the timestamps (see the README) |
| `status_raw` | VARCHAR | yes | Status exactly as in the source file |
| `actual_delay_hours` | DOUBLE | yes | `actual_arrival − planned_arrival` in hours. Null-safe. |
| `on_time_flag` | BOOLEAN | yes | `actual_delay_hours ≤ 24`. NULL when the outcome is unknown. |
| `transit_days_planned` | DOUBLE | no | `planned_arrival − planned_departure` in days |
| `transit_days_actual` | DOUBLE | yes | `actual_arrival − actual_departure` in days |
| `booking_lead_days` | DOUBLE | no | `planned_departure − booking_date` in days |
| `dq_flags` | VARCHAR[] | no | Any of: `actual_departure_before_booking`, `actual_arrival_before_departure`, `cargo_type_missing`, `cargo_type_unrecognised`, `cargo_type_normalised`, `weight_missing`, `weight_non_positive`, `container_count_missing`, `container_count_zero`, `container_count_outlier`, `status_missing_or_unmapped`, `status_normalised`, `no_observed_outcome` |
| `_source_row` | BIGINT | no | Line number in `raw.shipments` (`_row_number`) |

## `curated.ports`: one row per port (reference data)

| Column | Type | Definition |
|---|---|---|
| `port_code` | VARCHAR | **PK**. UN/LOCODE-style, 5 letters. |
| `port_name`, `country`, `region`, `timezone` | VARCHAR | `country` for BEANR is filled from `ref.port_country_fixes` (flag `country_filled_from_fix_list`) |
| `avg_congestion_score` | DOUBLE | 0–1 |
| `dq_flags` | VARCHAR[] | Fixes applied |

## `curated.port_events`: one row per distinct event

| Column | Type | Definition |
|---|---|---|
| `event_key` | VARCHAR | **PK**. md5 of all source fields. Deterministic, so re-runs produce the same key. |
| `event_id` | VARCHAR | Source id. **Not unique**: 103 ids are reused by different events (flag `event_id_collision`). |
| `event_timestamp` | TIMESTAMP | Event time |
| `port_code` | VARCHAR | **FK → `curated.ports`** |
| `vessel_id` | VARCHAR | Sentinel ids (VSL-0000 / VSL-9999) are quarantined |
| `event_type` | VARCHAR | One of ARRIVAL, BERTHED, CUSTOMS_CLEARED, DELAYED, DEPARTURE, INSPECTION, LOADING_COMPLETE |
| `delay_minutes` | INTEGER | 0–300 observed |
| `notes` | VARCHAR | Free text; NULL if blank. Not reliable (see DQ check `notes_contradict_delay`). |
| `dq_flags` | VARCHAR[] | `event_id_collision`, `delayed_event_without_delay` |
| `_source_row` | BIGINT | Line number in `raw.port_events` |

## `quarantine.*`

These tables hold the same columns as the typed stage, with the suffix `_raw` where curation would have changed the value, plus:

| Column | Definition |
|---|---|
| `quarantine_reason` | **shipments:** `conflicting_duplicate_id`, `unknown_port_code`, `same_origin_destination`, `missing_or_invalid_planned_timestamp`, `invalid_planned_sequence`. **events:** `unparseable_timestamp`, `missing_event_type`, `invalid_event_type`, `sentinel_vessel_id`, `unknown_port_code`. |
| `_row_number` | Line in the raw table |

## `raw.*`

Every source column is VARCHAR, exactly as in the file. Lineage columns: `_row_number` (1-based line), `_source_file`, `_source_sha256` and `_ingested_at`.

## `serving.*` (views)

| View | Columns |
|---|---|
| `route_stats` | `origin_port`, `destination_port`, `route_key`, `shipment_count`, `completed_count`, `cancelled_count`, `avg_delay_hours`, `median_delay_hours`, `p90_delay_hours`, `on_time_rate`, `first_planned_departure`, `last_planned_departure` |
| `route_quarterly_stats` | `origin_port`, `destination_port`, `route_key`, `year`, `quarter`, `period` (e.g. `2025-Q3`), `shipment_count`, `completed_count`, `avg_delay_hours`, `on_time_rate`. Quarters are by `planned_departure`. |
| `port_daily_activity` | `port_code`, `event_date`, `event_count`, `delayed_event_count`, `total_delay_minutes` |
| `shipments_enriched` | All `curated.shipments` columns (except lineage), plus origin and destination port name, country, region and congestion |

`on_time_rate` is always computed over **completed** shipments only, meaning those with an observed arrival.

## `ref.*` and `meta.*`

| Table | Contents |
|---|---|
| `ref.cargo_types`, `ref.status_aliases`, `ref.event_types`, `ref.sentinel_vessels`, `ref.port_country_fixes` | Generated from `shared/reference.py` on every run |
| `meta.pipeline_run` | `run_id`, `started_at`, `finished_at`, `pipeline_version`, `row_counts` (JSON), `source_files` (JSON with SHA-256 values), `content_fingerprint` |
| `meta.validation_results` | `rule`, `description`, `violations`. All 0 for a published warehouse. |
| `meta.table_fingerprints` | `relation`, `row_count`, `md5` for each deterministic relation |
