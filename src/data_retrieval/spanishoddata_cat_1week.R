# Paquetes necesarios
install.packages(c("spanishoddata", "dplyr", "purrr", "sf", "readr"))
library(spanishoddata)
library(dplyr)
library(purrr)
library(sf)
library(readr)

spod_set_data_dir("C:/temp/spanish_od_data")

# 1) Descargar zonas de MUNICIPIOS (v2) con geometría
#    Esto devuelve un objeto sf con un 'id' por municipio (clave que usa spanishoddata).
municipalities_sf <- spod_get_zones(zones = "municipalities", ver = 2)

# 2) Identificar municipios de Cataluña.
#    Estrategia: derivar el código INE de provincia a partir del identificador municipal.
#    Nota: En muchos datasets el identificador municipal (INE) empieza por los 2 dígitos de provincia:
#          Barcelona=08, Girona=17, Lleida=25, Tarragona=43.
#    Si tu versión de datos usa otra columna para la provincia, sustituye esta lógica por esa columna.
municipalities_sf <- municipalities_sf |>
  mutate(
    inferred_municipality_ine = sub(";.*$", "", municipalities), # si hay lista separada por ';', tomar el primero
    inferred_province_ine     = substr(inferred_municipality_ine, 1, 2)
  )

catalonia_province_codes <- c("08", "17", "25", "43")

catalonia_municipalities_sf <- municipalities_sf |>
  filter(inferred_province_ine %in% catalonia_province_codes) |>
  select(id, name, geometry)

catalonia_municipality_ids <- catalonia_municipalities_sf$id

# 3) Calcular centroides para asignar latitud/longitud a origen y destino.
#    Convertimos la geometría a WGS84 (EPSG:4326) y extraemos lon/lat en columnas simples.
municipality_centroids <- catalonia_municipalities_sf |>
  st_make_valid() |>
  st_centroid() |>
  st_transform(4326) |>
  mutate(
    longitude = st_coordinates(geometry)[, 1],
    latitude  = st_coordinates(geometry)[, 2]
  ) |>
  st_drop_geometry() |>
  select(id, name, longitude, latitude)

# 4) Rango de fechas: del 2024-01-01 al 2024-01-07 (ambos inclusive)
date_range <- seq(as.Date("2024-01-01"), as.Date("2024-01-07"), by = "day")

# 5) Descargar OD diario (municipio↔municipio) para Cataluña usando la vía "rápida".
#    'spod_quick_get_od()' ya devuelve datos agregados POR DÍA (no por hora), con:
#    date, id_origin, id_destination, n_trips, trips_total_length_km.
download_od_for_date <- function(single_date) {
  spod_quick_get_od(
    date           = single_date,
    min_trips      = 0,  # incluye pares con pocos viajes
    id_origin      = catalonia_municipality_ids,
    id_destination = catalonia_municipality_ids
  )
}

od_by_day_raw <- map_dfr(date_range, download_od_for_date)

# 6) Enriquecer con nombres y coordenadas (centroides) de origen y destino.
od_by_day_enriched <- od_by_day_raw |>
  rename(origin_id = id_origin, destination_id = id_destination) |>
  left_join(municipality_centroids, by = c("origin_id" = "id")) |>
  rename(
    origin_name      = name,
    origin_longitude = longitude,
    origin_latitude  = latitude
  ) |>
  left_join(municipality_centroids, by = c("destination_id" = "id")) |>
  rename(
    destination_name      = name,
    destination_longitude = longitude,
    destination_latitude  = latitude
  ) |>
  relocate(
    date,
    origin_id,   origin_name,   origin_longitude,   origin_latitude,
    destination_id, destination_name, destination_longitude, destination_latitude,
    n_trips, trips_total_length_km
  )

# 7) (Opcional) Reagrupar por día y par OD para garantizar unicidad si has hecho transformaciones.
od_by_day_and_pair <- od_by_day_enriched |>
  group_by(
    date,
    origin_id, origin_name, origin_longitude, origin_latitude,
    destination_id, destination_name, destination_longitude, destination_latitude
  ) |>
  summarise(
    n_trips = sum(n_trips),
    trips_total_length_km = sum(trips_total_length_km),
    .groups = "drop"
  )

# 8) Guardar a CSV
output_csv_path <- "C:/temp/od_catalonia_2024-01-01_2024-01-07.csv"
write_csv(od_by_day_and_pair, output_csv_path)

message("Archivo guardado: ", output_csv_path)
