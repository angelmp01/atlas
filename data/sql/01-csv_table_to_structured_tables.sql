-- ========================================
-- ARQUITECTURA NORMALIZADA - CONCLUSIONES
-- ========================================

-- CONCLUSIONES DEL ANÁLISIS:
-- 1. ZONAS ORIGEN: 14 únicos (relación 1:1 código-nombre, coordenadas consistentes)
-- 2. ZONAS DESTINO: 1113 únicos (relación 1:1 código-nombre, coordenadas consistentes)
-- 3. TIPO MERCANCIA: 2 tipos ("normal", "refrigerada")
-- 4. Las coordenadas son SIEMPRE iguales por zona (sin inconsistencias)
-- 5. No hay registros duplicados en origen/destino

-- ESTRUCTURA PROPUESTA:
-- ├── tipos_mercancia (pequeña tabla de referencia)
-- ├── zonas (consolidada con origen y destino)
-- ├── cargas_rutas (hechos con referencias a zonas y tipos)
-- └── cargas_rutas_geom (geometrías separadas opcionales)

-- ========================================
-- 1. TABLA: TIPOS_MERCANCIA
-- ========================================
CREATE TABLE IF NOT EXISTS app.tipos_mercancia (
    id_tipo_mercancia SMALLINT PRIMARY KEY,
    nombre_tipo VARCHAR(50) NOT NULL UNIQUE,
    descripcion VARCHAR(255)
);

INSERT INTO app.tipos_mercancia (id_tipo_mercancia, nombre_tipo, descripcion) VALUES
(1, 'normal', 'Mercancía estándar sin requerimientos especiales'),
(2, 'refrigerada', 'Mercancía que requiere cadena de frío');

-- ========================================
-- 2. TABLA: ZONAS (consolidada)
-- ========================================
-- Consolidamos origen y destino en una única tabla
CREATE TABLE IF NOT EXISTS app.zonas (
    id_zona SERIAL PRIMARY KEY,
    cod_zona VARCHAR(20) NOT NULL UNIQUE,
    nombre_zona VARCHAR(255) NOT NULL,
    longitude DOUBLE PRECISION,
    latitude DOUBLE PRECISION,
    geom geometry(Point, 25830),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Índices para búsquedas rápidas
CREATE INDEX idx_zonas_cod_zona ON app.zonas(cod_zona);
CREATE INDEX idx_zonas_nombre ON app.zonas(nombre_zona);
CREATE INDEX idx_zonas_geom ON app.zonas USING GIST(geom);

-- ========================================
-- 3. TABLA PRINCIPAL: CARGAS_RUTAS
-- ========================================
-- Opción recomendada: BIGSERIAL (autoincremental)
-- Razones:
-- - Genera IDs secuenciales (cache-friendly en B-tree)
-- - Más pequeño que UUID (8 bytes vs 16 bytes)
-- - Con 15M registros, no hay riesgo de overflow (2^63 - 1)
-- - Mejor rendimiento en índices y JOINs
-- - Las inserciones son muy rápidas sin contención

CREATE TABLE IF NOT EXISTS app.cargas_rutas (
    id_carga BIGSERIAL PRIMARY KEY,
    objectid BIGINT NOT NULL,  -- Identificador del CSV original (puede repetirse)
    fecha DATE NOT NULL,
    dia_semana VARCHAR(20) NOT NULL,
    id_zona_origen INTEGER NOT NULL,
    id_zona_destino INTEGER NOT NULL,
    id_tipo_mercancia SMALLINT NOT NULL,
    volumen DOUBLE PRECISION,
    peso DOUBLE PRECISION,
    precio DOUBLE PRECISION,
    geom_trayecto geometry(LineString, 25830),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_origen FOREIGN KEY (id_zona_origen) REFERENCES app.zonas(id_zona),
    CONSTRAINT fk_destino FOREIGN KEY (id_zona_destino) REFERENCES app.zonas(id_zona),
    CONSTRAINT fk_tipo_mercancia FOREIGN KEY (id_tipo_mercancia) REFERENCES app.tipos_mercancia(id_tipo_mercancia)
);

-- Índices para optimizar consultas
CREATE INDEX idx_cargas_objectid ON app.cargas_rutas(objectid);
CREATE INDEX idx_cargas_fecha ON app.cargas_rutas(fecha);
CREATE INDEX idx_cargas_zona_origen ON app.cargas_rutas(id_zona_origen);
CREATE INDEX idx_cargas_zona_destino ON app.cargas_rutas(id_zona_destino);
CREATE INDEX idx_cargas_tipo ON app.cargas_rutas(id_tipo_mercancia);
CREATE INDEX idx_cargas_fecha_origen_destino ON app.cargas_rutas(fecha, id_zona_origen, id_zona_destino);

-- ========================================
-- SCRIPT DE MIGRACIÓN (ejemplo simplificado)
-- ========================================

-- Paso 1: Insertar zonas origen
INSERT INTO app.zonas (cod_zona, nombre_zona, longitude, latitude)
SELECT DISTINCT cod_zona_origen, nombre_zona_origen, origen_longitude, origen_latitude
FROM app.cargas
WHERE cod_zona_origen IS NOT NULL
ON CONFLICT (cod_zona) DO NOTHING;

-- Paso 2: Insertar zonas destino
INSERT INTO app.zonas (cod_zona, nombre_zona, longitude, latitude)
SELECT DISTINCT cod_zona_destino, nombre_zona_destino, destino_longitude, destino_latitude
FROM app.cargas
WHERE cod_zona_destino IS NOT NULL
ON CONFLICT (cod_zona) DO NOTHING;

-- Paso 3: Actualizar geometrías
UPDATE app.zonas 
SET geom = ST_Point(longitude, latitude, 25830)
WHERE geom IS NULL;

-- Paso 4: Insertar datos en cargas_rutas (con parallelización recomendada)
INSERT INTO app.cargas_rutas 
(objectid, fecha, dia_semana, id_zona_origen, id_zona_destino, 
 id_tipo_mercancia, volumen, peso, precio, geom_trayecto)
SELECT 
    c.objectid,
    c.fecha,
    c.dia_semana,
    zo.id_zona,
    zd.id_zona,
    CASE WHEN c.tipo_mercancia = 'refrigerada' THEN 2 ELSE 1 END,
    c.volumen,
    c.peso,
    c.precio,
    c.geom_trayecto
FROM app.cargas c
JOIN app.zonas zo ON c.cod_zona_origen = zo.cod_zona
JOIN app.zonas zd ON c.cod_zona_destino = zd.cod_zona;

-- ========================================
-- QUERIES ÚTILES DESPUÉS DE LA MIGRACIÓN
-- ========================================

-- Verificar integridad
SELECT COUNT(*) as total_cargas FROM app.cargas_rutas;
SELECT COUNT(*) as total_zonas FROM app.zonas;

-- Rutas más frecuentes
SELECT 
    zo.nombre_zona as origen,
    zd.nombre_zona as destino,
    COUNT(*) as num_viajes,
    ROUND(AVG(cr.peso), 2) as peso_promedio,
    ROUND(AVG(cr.volumen), 2) as volumen_promedio
FROM app.cargas_rutas cr
JOIN app.zonas zo ON cr.id_zona_origen = zo.id_zona
JOIN app.zonas zd ON cr.id_zona_destino = zd.id_zona
GROUP BY cr.id_zona_origen, cr.id_zona_destino
ORDER BY num_viajes DESC
LIMIT 20;

-- Análisis por tipo de mercancía
SELECT 
    tm.nombre_tipo,
    COUNT(*) as num_cargas,
    ROUND(AVG(cr.peso), 2) as peso_promedio,
    ROUND(SUM(cr.precio), 0) as ingresos_totales
FROM app.cargas_rutas cr
JOIN app.tipos_mercancia tm ON cr.id_tipo_mercancia = tm.id_tipo_mercancia
GROUP BY cr.id_tipo_mercancia, tm.nombre_tipo;

-- Distancia media por ruta (usando geometrías)
SELECT 
    zo.nombre_zona as origen,
    zd.nombre_zona as destino,
    ROUND(ST_Distance(zo.geom, zd.geom) / 1000.0, 2) as distancia_km,
    COUNT(*) as num_viajes
FROM app.cargas_rutas cr
JOIN app.zonas zo ON cr.id_zona_origen = zo.id_zona
JOIN app.zonas zd ON cr.id_zona_destino = zd.id_zona
GROUP BY cr.id_zona_origen, cr.id_zona_destino, zo.geom, zd.geom
ORDER BY num_viajes DESC
LIMIT 20;