"""SQL for each pipeline layer.

Every statement is ``CREATE OR REPLACE`` and the whole build happens inside a
fresh database file, so re-running the pipeline is idempotent by construction.
"""

from __future__ import annotations

TS_FMT = "%Y-%m-%d %H:%M:%S"

SCHEMAS = "CREATE SCHEMA IF NOT EXISTS {}"
SCHEMA_NAMES = ("raw", "ref", "curated", "quarantine", "serving", "meta")

# --------------------------------------------------------------------------- #
# RAW — exact copy of the files, every column VARCHAR, plus lineage columns.
# --------------------------------------------------------------------------- #
RAW_LOAD = """
CREATE OR REPLACE TABLE raw.{table} AS
SELECT
    src.*,
    row_number() OVER () AS _row_number,
    ? AS _source_file,
    ? AS _source_sha256,
    CAST(? AS TIMESTAMP) AS _ingested_at
FROM read_csv(?, header = true, all_varchar = true) AS src
"""

# --------------------------------------------------------------------------- #
# CURATED — ports
# --------------------------------------------------------------------------- #
CURATED_PORTS = """
CREATE OR REPLACE TABLE curated.ports AS
SELECT
    upper(trim(p.port_code))                                  AS port_code,
    trim(p.port_name)                                         AS port_name,
    coalesce(nullif(trim(p.country), ''), f.country)          AS country,
    trim(p.region)                                            AS region,
    trim(p.timezone)                                          AS timezone,
    try_cast(p.avg_congestion_score AS DOUBLE)                AS avg_congestion_score,
    list_filter(
        [CASE WHEN nullif(trim(p.country), '') IS NULL AND f.country IS NOT NULL THEN 'country_filled_from_fix_list' END],
        x -> x IS NOT NULL
    )                                                         AS dq_flags
FROM raw.ports AS p
LEFT JOIN ref.port_country_fixes AS f ON f.port_code = upper(trim(p.port_code))
QUALIFY row_number() OVER (PARTITION BY upper(trim(p.port_code)) ORDER BY p._row_number) = 1
"""

# --------------------------------------------------------------------------- #
# CURATED — shipments
# Stage 1: drop exact duplicates, type every column, rank conflicting ids,
#          and decide a quarantine reason (NULL = row is good).
# --------------------------------------------------------------------------- #
STAGE_SHIPMENTS = f"""
CREATE OR REPLACE TEMP TABLE stg_shipments AS
WITH dedup AS (
    SELECT * FROM raw.shipments
    QUALIFY row_number() OVER (
        PARTITION BY shipment_id, customer_id, origin_port, destination_port, vessel_id, booking_date,
                     planned_departure, actual_departure, planned_arrival, actual_arrival,
                     container_count, cargo_type, weight_tons, status
        ORDER BY _row_number
    ) = 1
),
typed AS (
    SELECT
        trim(shipment_id)                                 AS shipment_id,
        trim(customer_id)                                 AS customer_id,
        upper(trim(origin_port))                          AS origin_port,
        upper(trim(destination_port))                     AS destination_port,
        trim(vessel_id)                                   AS vessel_id,
        try_strptime(booking_date, '{TS_FMT}')            AS booking_date,
        try_strptime(planned_departure, '{TS_FMT}')       AS planned_departure,
        try_strptime(actual_departure, '{TS_FMT}')        AS actual_departure_raw,
        try_strptime(planned_arrival, '{TS_FMT}')         AS planned_arrival,
        try_strptime(actual_arrival, '{TS_FMT}')          AS actual_arrival_raw,
        try_cast(container_count AS INTEGER)              AS container_count_raw,
        try_cast(weight_tons AS DOUBLE)                   AS weight_tons_raw,
        cargo_type                                        AS cargo_type_raw,
        status                                            AS status_raw,
        _row_number,
        ( (customer_id IS NULL)::INT + (vessel_id IS NULL)::INT + (booking_date IS NULL)::INT
        + (planned_departure IS NULL)::INT + (actual_departure IS NULL)::INT
        + (planned_arrival IS NULL)::INT + (actual_arrival IS NULL)::INT
        + (container_count IS NULL)::INT + (nullif(trim(cargo_type), '') IS NULL)::INT
        + (weight_tons IS NULL)::INT + (nullif(trim(status), '') IS NULL)::INT ) AS blank_field_count
    FROM dedup
),
ranked AS (
    SELECT *,
        row_number() OVER (PARTITION BY shipment_id ORDER BY blank_field_count, _row_number) AS id_rank
    FROM typed
)
SELECT
    r.*,
    CASE
        WHEN r.id_rank > 1                                                     THEN 'conflicting_duplicate_id'
        WHEN o.port_code IS NULL OR d.port_code IS NULL                        THEN 'unknown_port_code'
        WHEN r.origin_port = r.destination_port                                THEN 'same_origin_destination'
        WHEN r.booking_date IS NULL OR r.planned_departure IS NULL
          OR r.planned_arrival IS NULL                                         THEN 'missing_or_invalid_planned_timestamp'
        WHEN r.planned_arrival <= r.planned_departure
          OR r.planned_departure < r.booking_date                              THEN 'invalid_planned_sequence'
    END AS quarantine_reason
FROM ranked AS r
LEFT JOIN curated.ports AS o ON o.port_code = r.origin_port
LEFT JOIN curated.ports AS d ON d.port_code = r.destination_port
"""

QUARANTINE_SHIPMENTS = """
CREATE OR REPLACE TABLE quarantine.shipments AS
SELECT * EXCLUDE (id_rank, blank_field_count) FROM stg_shipments WHERE quarantine_reason IS NOT NULL
"""

# Stage 2: clean the surviving rows and derive the curated fields.
CURATED_SHIPMENTS = """
CREATE OR REPLACE TABLE curated.shipments AS
WITH base AS (
    SELECT s.*,
        sa.canonical_status,
        ct.canonical_cargo_type,
        -- an actual departure before booking is impossible -> discard it
        (s.actual_departure_raw < s.booking_date)                             AS bad_actual_departure,
        (s.actual_arrival_raw <= s.actual_departure_raw)                      AS bad_actual_sequence
    FROM stg_shipments AS s
    LEFT JOIN ref.status_aliases AS sa ON sa.raw_status = upper(trim(s.status_raw))
    LEFT JOIN ref.cargo_types   AS ct ON ct.match_key = lower(trim(regexp_replace(s.cargo_type_raw, '\\s+', ' ', 'g')))
    WHERE s.quarantine_reason IS NULL
),
cleaned AS (
    SELECT *,
        CASE WHEN coalesce(bad_actual_departure, false) OR coalesce(bad_actual_sequence, false)
             THEN NULL ELSE actual_departure_raw END                         AS actual_departure,
        CASE WHEN coalesce(bad_actual_sequence, false)
             THEN NULL ELSE actual_arrival_raw END                           AS actual_arrival
    FROM base
),
derived AS (
    SELECT *,
        date_diff('minute', planned_arrival, actual_arrival) / 60.0          AS actual_delay_hours
    FROM cleaned
)
SELECT
    shipment_id,
    customer_id,
    origin_port,
    destination_port,
    origin_port || ' → ' || destination_port                                AS route_key,
    vessel_id,
    booking_date,
    planned_departure,
    actual_departure,
    planned_arrival,
    actual_arrival,
    CASE WHEN container_count_raw BETWEEN 1 AND $max_containers
         THEN container_count_raw END                                        AS container_count,
    coalesce(canonical_cargo_type, $unknown_cargo)                           AS cargo_type,
    CASE WHEN weight_tons_raw > 0 THEN weight_tons_raw END                   AS weight_tons,
    CASE
        WHEN canonical_status = 'CANCELLED' AND actual_arrival IS NULL       THEN 'CANCELLED'
        WHEN actual_arrival IS NOT NULL AND canonical_status IS DISTINCT FROM 'CANCELLED'
             THEN CASE WHEN actual_delay_hours > $on_time_hours THEN 'DELAYED' ELSE 'DELIVERED' END
        ELSE $unknown_status
    END                                                                      AS status,
    status_raw,
    actual_delay_hours,
    CASE WHEN actual_delay_hours IS NOT NULL
         THEN actual_delay_hours <= $on_time_hours END                       AS on_time_flag,
    date_diff('minute', planned_departure, planned_arrival) / 1440.0         AS transit_days_planned,
    date_diff('minute', actual_departure, actual_arrival) / 1440.0           AS transit_days_actual,
    date_diff('minute', booking_date, planned_departure) / 1440.0            AS booking_lead_days,
    list_filter([
        CASE WHEN coalesce(bad_actual_departure, false)                       THEN 'actual_departure_before_booking' END,
        CASE WHEN coalesce(bad_actual_sequence, false)                        THEN 'actual_arrival_before_departure' END,
        CASE WHEN cargo_type_raw IS NULL OR trim(cargo_type_raw) = ''         THEN 'cargo_type_missing'
             WHEN canonical_cargo_type IS NULL                                THEN 'cargo_type_unrecognised'
             WHEN cargo_type_raw <> canonical_cargo_type                      THEN 'cargo_type_normalised' END,
        CASE WHEN weight_tons_raw IS NULL                                     THEN 'weight_missing'
             WHEN weight_tons_raw <= 0                                        THEN 'weight_non_positive' END,
        CASE WHEN container_count_raw IS NULL                                 THEN 'container_count_missing'
             WHEN container_count_raw <= 0                                    THEN 'container_count_zero'
             WHEN container_count_raw > $max_containers                       THEN 'container_count_outlier' END,
        CASE WHEN canonical_status IS NULL                                    THEN 'status_missing_or_unmapped'
             WHEN status_raw <> canonical_status                              THEN 'status_normalised' END,
        CASE WHEN actual_arrival IS NULL AND canonical_status IS DISTINCT FROM 'CANCELLED'
             THEN 'no_observed_outcome' END
    ], x -> x IS NOT NULL)                                                   AS dq_flags
FROM derived
ORDER BY shipment_id
"""

# --------------------------------------------------------------------------- #
# CURATED — port events (streaming-style feed)
# --------------------------------------------------------------------------- #
STAGE_EVENTS = f"""
CREATE OR REPLACE TEMP TABLE stg_port_events AS
WITH dedup AS (
    SELECT * FROM raw.port_events
    QUALIFY row_number() OVER (
        PARTITION BY event_id, event_timestamp, port_code, vessel_id, event_type, delay_minutes, notes
        ORDER BY _row_number
    ) = 1
)
SELECT
    md5(concat_ws('|', e.event_id, e.event_timestamp, e.port_code, e.vessel_id,
                       e.event_type, e.delay_minutes, coalesce(e.notes, '')))   AS event_key,
    trim(e.event_id)                                                         AS event_id,
    try_strptime(e.event_timestamp, '{TS_FMT}')                              AS event_timestamp,
    upper(trim(e.port_code))                                                 AS port_code,
    trim(e.vessel_id)                                                        AS vessel_id,
    nullif(upper(trim(e.event_type)), '')                                    AS event_type,
    try_cast(e.delay_minutes AS INTEGER)                                     AS delay_minutes,
    nullif(trim(e.notes), '')                                                AS notes,
    count(*) OVER (PARTITION BY trim(e.event_id)) > 1                        AS event_id_collision,
    CASE
        WHEN try_strptime(e.event_timestamp, '{TS_FMT}') IS NULL            THEN 'unparseable_timestamp'
        WHEN nullif(trim(e.event_type), '') IS NULL                          THEN 'missing_event_type'
        WHEN upper(trim(e.event_type)) NOT IN (SELECT event_type FROM ref.event_types)
                                                                             THEN 'invalid_event_type'
        WHEN trim(e.vessel_id) IN (SELECT vessel_id FROM ref.sentinel_vessels) THEN 'sentinel_vessel_id'
        WHEN p.port_code IS NULL                                             THEN 'unknown_port_code'
    END                                                                      AS quarantine_reason,
    e._row_number
FROM dedup AS e
LEFT JOIN curated.ports AS p ON p.port_code = upper(trim(e.port_code))
"""

QUARANTINE_EVENTS = """
CREATE OR REPLACE TABLE quarantine.port_events AS
SELECT * FROM stg_port_events WHERE quarantine_reason IS NOT NULL
"""

CURATED_EVENTS = """
CREATE OR REPLACE TABLE curated.port_events AS
SELECT
    event_key, event_id, event_timestamp, port_code, vessel_id, event_type, delay_minutes, notes,
    list_filter([
        CASE WHEN event_id_collision                          THEN 'event_id_collision' END,
        CASE WHEN event_type = 'DELAYED' AND delay_minutes = 0 THEN 'delayed_event_without_delay' END
    ], x -> x IS NOT NULL) AS dq_flags
FROM stg_port_events
WHERE quarantine_reason IS NULL
ORDER BY event_timestamp, event_key
"""

# --------------------------------------------------------------------------- #
# SERVING — consumer-shaped views on top of curated
# --------------------------------------------------------------------------- #
SERVING_ROUTE_STATS = """
CREATE OR REPLACE VIEW serving.route_stats AS
SELECT
    origin_port,
    destination_port,
    route_key,
    count(*)                                                AS shipment_count,
    count(actual_delay_hours)                               AS completed_count,
    count(*) FILTER (WHERE status = 'CANCELLED')            AS cancelled_count,
    avg(actual_delay_hours)                                 AS avg_delay_hours,
    median(actual_delay_hours)                              AS median_delay_hours,
    quantile_cont(actual_delay_hours, 0.9)                  AS p90_delay_hours,
    avg(CASE WHEN on_time_flag THEN 1.0 WHEN NOT on_time_flag THEN 0.0 END) AS on_time_rate,
    min(planned_departure)                                  AS first_planned_departure,
    max(planned_departure)                                  AS last_planned_departure
FROM curated.shipments
GROUP BY ALL
"""

SERVING_PORT_DAILY = """
CREATE OR REPLACE VIEW serving.port_daily_activity AS
SELECT
    port_code,
    CAST(event_timestamp AS DATE)                           AS event_date,
    count(*)                                                AS event_count,
    count(*) FILTER (WHERE event_type = 'DELAYED')          AS delayed_event_count,
    sum(delay_minutes)                                      AS total_delay_minutes
FROM curated.port_events
GROUP BY ALL
"""

PIPELINE_RUN = """
CREATE OR REPLACE TABLE meta.pipeline_run AS
SELECT
    CAST(? AS VARCHAR)   AS run_id,
    CAST(? AS TIMESTAMP) AS started_at,
    CAST(? AS TIMESTAMP) AS finished_at,
    CAST(? AS VARCHAR)   AS pipeline_version,
    CAST(? AS JSON)      AS row_counts,
    CAST(? AS JSON)      AS source_files
"""
