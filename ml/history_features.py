"""Point-in-time historical features — evaluated as an experiment, NOT used in production.

Each feature for a shipment uses only information that existed *before its booking_date*:
outcomes of shipments that had already arrived, and port events already logged. This is
the leakage-safe way to use history (a naive group-by over the whole table would leak the
future into the past).

They were tested because "this vessel / customer / route has been late recently" is the
most plausible source of signal. Result: no lift (see model_metadata.json → experiments),
and serving them would need a feature store. So they stay out of the production model.
"""

from __future__ import annotations

import duckdb
import pandas as pd

HISTORY_FEATURES: tuple[str, ...] = (
    "hist_vessel_late_rate",
    "hist_customer_late_rate",
    "hist_route_late_rate",
    "hist_origin_late_rate",
    "hist_origin_event_delay_30d",
    "hist_origin_delayed_events_30d",
    "hist_destination_event_delay_30d",
)

_SQL = """
WITH done AS (
    SELECT shipment_id, vessel_id, customer_id, route_key, origin_port, actual_arrival,
           (actual_delay_hours > 24)::INT AS late
    FROM curated.shipments WHERE actual_delay_hours IS NOT NULL
)
SELECT
    s.shipment_id,
    (SELECT avg(h.late) FROM done h WHERE h.vessel_id = s.vessel_id AND h.actual_arrival < s.booking_date)
        AS hist_vessel_late_rate,
    (SELECT avg(h.late) FROM done h WHERE h.customer_id = s.customer_id AND h.actual_arrival < s.booking_date)
        AS hist_customer_late_rate,
    (SELECT avg(h.late) FROM done h WHERE h.route_key = s.route_key AND h.actual_arrival < s.booking_date)
        AS hist_route_late_rate,
    (SELECT avg(h.late) FROM done h WHERE h.origin_port = s.origin_port AND h.actual_arrival < s.booking_date)
        AS hist_origin_late_rate,
    (SELECT avg(e.delay_minutes) FROM curated.port_events e WHERE e.port_code = s.origin_port
        AND e.event_timestamp < s.booking_date AND e.event_timestamp >= s.booking_date - INTERVAL 30 DAY)
        AS hist_origin_event_delay_30d,
    (SELECT count(*) FILTER (WHERE e.event_type = 'DELAYED') FROM curated.port_events e
        WHERE e.port_code = s.origin_port
        AND e.event_timestamp < s.booking_date AND e.event_timestamp >= s.booking_date - INTERVAL 30 DAY)
        AS hist_origin_delayed_events_30d,
    (SELECT avg(e.delay_minutes) FROM curated.port_events e WHERE e.port_code = s.destination_port
        AND e.event_timestamp < s.booking_date AND e.event_timestamp >= s.booking_date - INTERVAL 30 DAY)
        AS hist_destination_event_delay_30d
FROM curated.shipments AS s
WHERE s.actual_delay_hours IS NOT NULL
"""


def load_history_features(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(_SQL).df().set_index("shipment_id")
