# ATLAS Web Platform

Sistema web para optimización logística de transporte por carretera, desarrollado como proyecto académico.

## 🏗️ Arquitectura

El proyecto está dividido en dos módulos independientes:

### **atlas_api/** - Backend API REST
- Proporciona endpoints para datos de localizaciones, tipos de mercancía y cálculo de rutas
- Tecnología: FastAPI + PostgreSQL
- Puerto por defecto: `8000`

### **atlas_web/** - Frontend Web
- Interfaz web con formulario y mapa interactivo
- Consume la API REST para obtener datos
- Tecnología: FastAPI + Jinja2 + Leaflet.js
- Puerto por defecto: `8001`

## 🚀 Inicio Rápido

### Requisitos Previos

- Python 3.8+
- PostgreSQL (opcional, puede usar datos mock)
- pip o conda para gestión de paquetes

### Instalación

1. **Clonar el repositorio** (si aplica):
```bash
cd atlas
```

2. **Instalar dependencias**:
```bash
pip install fastapi uvicorn jinja2 python-multipart sqlalchemy psycopg2-binary
```

O si usas conda:
```bash
conda install -c conda-forge fastapi uvicorn jinja2 python-multipart sqlalchemy psycopg2
```

### Ejecución

#### Opción 1: Ejecutar ambos servicios por separado (recomendado)

**Terminal 1 - API:**
```bash
python -m atlas_api.main
```

**Terminal 2 - Web:**
```bash
python -m atlas_web.main
```

#### Opción 2: Usar scripts de inicio

**Windows (PowerShell):**
```powershell
.\start_api.ps1
.\start_web.ps1
```

**Linux/Mac:**
```bash
./start_api.sh
./start_web.sh
```

### Acceso

- **Web Interface:** http://localhost:8001
- **API Docs:** http://localhost:8000/docs
- **API Health:** http://localhost:8000/health

## 📡 API Endpoints

### `GET /locations`
Devuelve lista de localizaciones disponibles.

**Respuesta:**
```json
[
  {
    "id": "08019",
    "name": "Barcelona",
    "latitude": 41.3851,
    "longitude": 2.1734
  }
]
```

### `GET /goods-types`
Devuelve tipos de mercancía disponibles.

**Respuesta:**
```json
[
  {
    "id": "normal",
    "name": "Normal",
    "description": "Mercancía estándar"
  },
  {
    "id": "refrigerada",
    "name": "Refrigerada",
    "description": "Mercancía refrigerada"
  }
]
```

### `GET /goods-types/truck-types`
Devuelve tipos de camión disponibles.

**Respuesta:**
```json
[
  {
    "id": "normal",
    "name": "Normal",
    "description": "Camión estándar"
  },
  {
    "id": "refrigerado",
    "name": "Refrigerado",
    "description": "Camión frigorífico"
  }
]
```

## 🌐 Interfaz Web

La interfaz web (`atlas_web`) incluye:

1. **Formulario de parámetros:**
   - Selección de origen y destino (cargado dinámicamente desde API)
   - Tipo de mercancía (cargado desde API)
   - Tipo de camión (cargado desde API)
   - Buffer de búsqueda en km
   - Fecha

2. **Mapa interactivo:**
   - Visualización de todas las localizaciones disponibles
   - Marcadores interactivos con información
   - Línea de ruta entre origen y destino (cuando se calcula)
   - Zoom automático a la ruta seleccionada

3. **Resultados:**
   - Muestra parámetros de búsqueda
   - Preparado para mostrar resultados de optimización (próximamente)

## ⚙️ Configuración

### Variables de Entorno

Puedes configurar los servicios mediante variables de entorno:

**API (`atlas_api`):**
```bash
API_HOST=127.0.0.1
API_PORT=8000
API_RELOAD=true
PG_DSN=postgresql://user:pass@host:port/db
```

**Web (`atlas_web`):**
```bash
WEB_HOST=127.0.0.1
WEB_PORT=8001
WEB_RELOAD=true
API_BASE_URL=http://127.0.0.1:8000
```

### Base de Datos

Por defecto, la API intenta conectarse a PostgreSQL usando la configuración en `atlas_api/config.py`.

Si la conexión falla, **automáticamente usa datos mock** (4 capitales de Cataluña) sin interrumpir el servicio.

Para usar la base de datos real, configura la variable `PG_DSN` con tu cadena de conexión.

## 🐛 Debugging

### API no responde

1. Verifica que el servidor está corriendo:
```bash
curl http://localhost:8000/health
```

2. Revisa los logs en la terminal donde ejecutaste la API

3. Verifica la configuración de CORS en `atlas_api/config.py`

### Web no carga datos

1. Abre la consola del navegador (F12) y busca errores
2. Verifica que la API está accesible desde el navegador
3. Comprueba que `API_BASE_URL` en `atlas_web/config.py` es correcta

### Errores de base de datos

Si ves errores relacionados con PostgreSQL:
- La API automáticamente usa datos mock
- Verifica la cadena de conexión en `atlas_api/config.py`
- Asegúrate de que la tabla `app.sodd_zones` existe si usas BD real

## 📂 Estructura del Proyecto

```
atlas/
├── atlas_api/              # API REST backend
│   ├── __init__.py
│   ├── main.py            # FastAPI app
│   ├── config.py          # Configuración
│   ├── routes/            # Endpoints
│   │   ├── locations.py
│   │   └── goods_types.py
│   └── data/
│       └── mock_data.py   # Datos mock/DB
│
├── atlas_web/             # Web frontend
│   ├── __init__.py
│   ├── main.py           # FastAPI app (servir HTML)
│   ├── config.py         # Configuración
│   ├── templates/
│   │   └── index.html    # Página principal
│   └── static/
│       ├── css/
│       │   └── style.css
│       └── js/
│           └── map.js    # Lógica de mapa y API calls
│
├── atlas_ml/              # ML models (existente)
├── models/                # Modelos entrenados
├── data/                  # Datasets
└── README_WEB.md         # Esta documentación
```

## 🔜 Próximas Funcionalidades

- [ ] Endpoint `/route` para calcular rutas básicas O→D
- [ ] Endpoint `/inference` para predicciones con modelos ML
- [ ] Visualización de rutas óptimas en el mapa
- [ ] Mostrar puntos intermedios de carga
- [ ] Estadísticas de la ruta (distancia, tiempo estimado, coste)
- [ ] Exportar resultados a CSV/JSON

## 🎓 Notas Académicas

Este es un proyecto académico con las siguientes características:

- ✅ Código simple y legible
- ✅ Errores mostrados claramente (sin ocultarlos)
- ✅ Sin autenticación (no necesaria para MVP)
- ✅ Sin tests automáticos (fase de prototipo)
- ✅ Configuración mínima viable
- ✅ Documentación práctica

## 📝 Licencia

Proyecto académico - Universidad Politécnica de Cataluña (UPC)

## 👥 Contacto

Para dudas o problemas, revisar los logs de consola donde se muestran todos los errores detalladamente.
