DROP MATERIALIZED VIEW IF EXISTS app.sodd_loads_filtered;

CREATE MATERIALIZED VIEW app.sodd_loads_filtered AS
SELECT
  l.*,
  (l.n_trips * 0.04::numeric) AS n_trips_logistic  -- factor “camiones” ≈ 4%
FROM app.sodd_loads l
WHERE (l.trips_total_length_km / NULLIF(l.n_trips, 0)) >= 10;


CREATE INDEX IF NOT EXISTS sodd_loads_filt_od_idx   ON app.sodd_loads_filtered (origin_id, destination_id);
CREATE INDEX IF NOT EXISTS sodd_loads_filt_date_idx ON app.sodd_loads_filtered (date);
CREATE INDEX IF NOT EXISTS sodd_loads_filt_tipo_idx ON app.sodd_loads_filtered (tipo_mercancia);
-- Si haces queries espaciales sobre geom:
CREATE INDEX IF NOT EXISTS sodd_loads_filt_geom_gix ON app.sodd_loads_filtered USING GIST (geom);

SELECT COUNT(*) AS total, ROUND(AVG(n_trips_logistic),1) AS avg_logistic
FROM app.sodd_loads_filtered;
