-- Asegura PostGIS (para la parte C)
CREATE EXTENSION IF NOT EXISTS postgis;

-- Tabla raw (solo datos del CSV, sin geom)
DROP TABLE IF EXISTS app.sodd_loadeddata;
CREATE TABLE app.sodd_loadeddata (
  date                      date,
  origin_id                 text,
  origin_name               text,
  origin_longitude          double precision, -- WGS84 lon
  origin_latitude           double precision, -- WGS84 lat
  destination_id            text,
  destination_name          text,
  destination_longitude     double precision,
  destination_latitude      double precision,
  n_trips                   integer,
  trips_total_length_km     numeric(12,3),
  tipo_mercancia            text,
  volumen                   integer,
  peso                      integer,
  precio                    integer
);


-- Tabla de localizaciones (orígenes y destinos únicos con geom)
DROP TABLE IF EXISTS app.sodd_locations;
CREATE TABLE app.sodd_locations (
  location_id   text PRIMARY KEY,
  location_name text,
  longitude     double precision,
  latitude      double precision,
  geom          geometry(Point, 4326)
);

-- Insertar orígenes y destinos, resolviendo por location_id
WITH origins AS (
  SELECT DISTINCT ON (origin_id)
         origin_id AS location_id,
         origin_name AS location_name,
         origin_longitude AS longitude,
         origin_latitude AS latitude
  FROM app.sodd_loadeddata
  ORDER BY origin_id
),
dests AS (
  SELECT DISTINCT ON (destination_id)
         destination_id AS location_id,
         destination_name AS location_name,
         destination_longitude AS longitude,
         destination_latitude AS latitude
  FROM app.sodd_loadeddata
  ORDER BY destination_id
),
unioned AS (
  SELECT * FROM origins
  UNION
  SELECT * FROM dests
)
INSERT INTO app.sodd_locations (location_id, location_name, longitude, latitude, geom)
SELECT
  location_id,
  location_name,
  longitude,
  latitude,
  ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)
FROM (
  -- en caso de duplicados por location_id en origins/dests, nos quedamos con la primera fila
  SELECT DISTINCT ON (location_id) *
  FROM unioned
  ORDER BY location_id
) t;

CREATE INDEX sodd_locations_name_idx ON app.sodd_locations (location_name);
CREATE INDEX sodd_locations_geom_gix ON app.sodd_locations USING GIST (geom);


-- Tabla final de cargas (unión de raw + localizaciones para crear geom)
-- NOTA: La tabla base sodd_loads contiene los datos originales del SpanishODData
-- La vista materializada sodd_loads_logistics (creada externamente) contiene
-- estimaciones logísticas con n_trips_logistic calculado a partir de volumen de camiones
DROP TABLE IF EXISTS app.sodd_loads;
CREATE TABLE app.sodd_loads (
  date                      date,
  origin_id                 text REFERENCES app.sodd_locations(location_id),
  destination_id            text REFERENCES app.sodd_locations(location_id),
  n_trips                   integer,
  trips_total_length_km     numeric(12,3),
  tipo_mercancia            text,
  volumen                   integer,
  peso                      integer,
  precio                    integer,
  geom                      geometry(LineString, 4326)
);

-- Poblar las cargas uniendo a las localizaciones para construir la línea
INSERT INTO app.sodd_loads (
  date, origin_id, destination_id,
  n_trips, trips_total_length_km, tipo_mercancia, volumen, peso, precio, geom
)
SELECT
  ld.date,
  ld.origin_id,
  ld.destination_id,
  ld.n_trips,
  ld.trips_total_length_km,
  ld.tipo_mercancia,
  ld.volumen,
  ld.peso,
  ld.precio,
  ST_MakeLine(lo.geom, ldv.geom)  -- LineString origen→destino
FROM app.sodd_loadeddata ld
JOIN app.sodd_locations lo  ON lo.location_id = ld.origin_id
JOIN app.sodd_locations ldv ON ldv.location_id = ld.destination_id;

-- Índices
CREATE INDEX sodd_loads_date_idx       ON app.sodd_loads (date);
CREATE INDEX sodd_loads_o_idx          ON app.sodd_loads (origin_id);
CREATE INDEX sodd_loads_d_idx          ON app.sodd_loads (destination_id);
CREATE INDEX sodd_loads_tipo_idx       ON app.sodd_loads (tipo_mercancia);
CREATE INDEX sodd_loads_geom_gix       ON app.sodd_loads USING GIST (geom);

VACUUM ANALYZE app.sodd_locations;
VACUUM ANALYZE app.sodd_loads;
