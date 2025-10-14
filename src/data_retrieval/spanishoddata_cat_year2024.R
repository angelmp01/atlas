# Paquetes necesarios
# install.packages(c("spanishoddata", "dplyr", "purrr", "sf", "readr"))
library(spanishoddata)
library(dplyr)
library(purrr)
library(sf)
library(readr)

spod_set_data_dir("C:/temp/spanish_od_data")

# ============================================================================
# CONFIGURACIÓN: CAMBIAR EL MES AQUÍ (01 a 12)
# ============================================================================
mes <- "01"  # Cambia este valor: "01", "02", "03", ..., "12"
ano <- 2024
# ============================================================================

# 1) Descargar zonas de MUNICIPIOS (v2) con geometría
municipalities_sf <- spod_get_zones(zones = "municipalities", ver = 2)

# 2) Identificar municipios de Cataluña
municipalities_sf <- municipalities_sf |>
  mutate(
    inferred_municipality_ine = sub(";.*$", "", municipalities),
    inferred_province_ine     = substr(inferred_municipality_ine, 1, 2)
  )

catalonia_province_codes <- c("08", "17", "25", "43")
catalonia_municipalities_sf <- municipalities_sf |>
  filter(inferred_province_ine %in% catalonia_province_codes) |>
  select(id, name, geom)  # ← Cambiar "geometry" por "geom"

catalonia_municipality_ids <- catalonia_municipalities_sf$id

# 3) Calcular centroides
municipality_centroids <- catalonia_municipalities_sf |>
  st_make_valid() |>
  st_centroid() |>
  st_transform(4326) |>
  mutate(
    longitude = st_coordinates(geom)[, 1],
    latitude  = st_coordinates(geom)[, 2]
  ) |>
  st_drop_geometry() |>
  select(id, name, longitude, latitude)

# 4) Crear rango de fechas SOLO PARA EL MES ESPECIFICADO
fecha_inicio <- as.Date(paste0(ano, "-", mes, "-01"))
# Último día del mes
fecha_fin <- as.Date(paste0(ano, "-", sprintf("%02d", as.numeric(mes) + 1), "-01")) - 1

# date_range <- seq(fecha_inicio, fecha_fin, by = "day")
es_fecha_valida <- function(fecha) {
  (fecha >= as.Date("2023-11-04") & fecha <= as.Date("2024-04-17")) |
    (fecha >= as.Date("2024-04-19") & fecha <= as.Date("2024-11-08")) |
    (fecha >= as.Date("2024-11-11") & fecha <= as.Date("2024-11-25")) |
    (fecha >= as.Date("2024-11-27") & fecha <= as.Date("2024-12-31")) |
    (fecha >= as.Date("2025-01-13") & fecha <= as.Date("2025-01-19")) |
    (fecha >= as.Date("2025-02-03") & fecha <= as.Date("2025-02-09")) |
    (fecha >= as.Date("2025-03-10") & fecha <= as.Date("2025-03-31")) |
    (fecha >= as.Date("2025-04-04") & fecha <= as.Date("2025-04-10")) |
    (fecha >= as.Date("2025-04-17") & fecha <= as.Date("2025-04-17")) |
    (fecha >= as.Date("2025-05-01") & fecha <= as.Date("2025-05-02")) |
    (fecha >= as.Date("2025-05-19") & fecha <= as.Date("2025-05-25")) |
    (fecha >= as.Date("2025-06-02") & fecha <= as.Date("2025-06-03")) |
    (fecha >= as.Date("2025-06-05") & fecha <= as.Date("2025-06-08"))
}

date_range <- seq(fecha_inicio, fecha_fin, by = "day") %>%
  keep(es_fecha_valida)




message("Descargando datos para el período: ", fecha_inicio, " al ", fecha_fin)
message("Total de días a descargar: ", length(date_range))

# 5) Descargar OD diario para el mes especificado
download_od_for_date <- function(single_date) {
  message("  Descargando: ", single_date)
  
  resultado <- spod_quick_get_od(
    date           = single_date,
    min_trips      = 0,
    id_origin      = catalonia_municipality_ids,
    id_destination = catalonia_municipality_ids
  )
  
  Sys.sleep(2)  # Esperar 2 segundos entre peticiones
  
  return(resultado)
}

od_by_day_raw <- map_dfr(date_range, download_od_for_date)

# 6) Enriquecer con nombres y coordenadas
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

# 7) Reagrupar por día y par OD para garantizar unicidad
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

# 8) Guardar a CSV con el mes en el nombre
output_csv_path <- sprintf("C:/temp/od_catalonia_%d_mes_%s.csv", ano, mes)
write_csv(od_by_day_and_pair, output_csv_path)

message("✓ Archivo guardado: ", output_csv_path)
message("  Número de filas: ", nrow(od_by_day_and_pair))
message("  Período: ", fecha_inicio, " a ", fecha_fin)
