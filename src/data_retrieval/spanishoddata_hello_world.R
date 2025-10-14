install.packages("rtools")
install.packages("spanishoddata")

library(spanishoddata)
library(dplyr)

# 2) (Opcional) Carpeta de trabajo para metadatos
spod_set_data_dir("C:/temp/spanish_od_data")

# 3) Zonas (municipios) y selección de Cataluña
muni <- spod_quick_get_zones()  # IDs y nombres a nivel municipal (v2 rápido)
# INE provincias Cataluña: Barcelona(08), Girona(17), Lleida(25), Tarragona(43)
cat_ids <- muni |> 
  filter(substr(id, 1, 2) %in% c("08","17","25","43")) |> 
  pull(id)

# 4) OD diario rápido (solo 1 día) y filtrado a Cataluña
od_cat <- spod_quick_get_od(
  date = "2024-01-01",   # elige la fecha que quieras disponible
  min_trips = 0,
  id_origin = cat_ids,
  id_destination = cat_ids
)

# 5) Top 10 relaciones OD dentro de Cataluña
od_cat |>
  arrange(desc(n_trips)) |>
  head(10)

