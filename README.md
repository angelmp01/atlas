# Atlas - Truck Delivery Data Analytics 🚛

Atlas is a post-graduate research project focused on analyzing truck delivery data to gain insights into transportation patterns and efficiency. The system processes raw delivery data and provides comprehensive analytics for understanding truck delivery operations.

## 🎯 Project Overview

The Atlas project focuses on processing and analyzing truck delivery data to understand transportation patterns. The system helps researchers and analysts:

- **Process diverse delivery data formats**
- **Standardize transportation datasets**
- **Analyze delivery patterns and efficiency**
- **Generate insights from transportation data**

## 🏗️ Project Structure

```
atlas/
├── src/                          # Source code
│   ├── preprocessing/            # Data preprocessing modules
│   │   ├── __init__.py
│   │   ├── data_preprocessor.py  # Main preprocessing class
│   │   └── main.py               # Preprocessing entry point
│   ├── utils/                    # Utility functions
│   │   ├── __init__.py
│   │   ├── logger.py             # Logging utilities
│   │   └── file_handler.py       # File handling utilities
│   ├── __init__.py
│   └── main.py                   # Main project entry point
├── data/                         # Data directories
│   ├── raw/                      # Raw input data files
│   ├── processed/                # Processed CSV outputs
│   └── interim/                  # Intermediate processing files
├── config/                       # Configuration files
│   └── config.json               # Main configuration
├── notebooks/                    # Jupyter notebooks for analysis
├── tests/                        # Unit tests
├── docs/                         # Documentation
├── docker/                       # Docker files to create DB
├── logs/                         # Log files (auto-generated)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- docker
- docker-compose 3.8

### Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd atlas
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run preprocessing**:
   ```bash
   python src/main.py preprocessing
   ```
   
   Or run the module directly:
   ```bash
   python src/preprocessing/main.py
   ```

### Basic Usage

### Basic Usage

#### Option 1: Using the main entry point (Recommended)
```bash
python src/main.py preprocessing
```

#### Option 2: Using the module directly
```bash
python src/preprocessing/main.py
```

#### Option 3: With custom parameters
```bash
# Basic preprocessing with exact distribution
python src/preprocessing/main.py --input-dir data/raw/hermes --output-dir data/processed --distribution exact

# With Poisson distribution for realistic variability
python src/preprocessing/main.py --input-dir data/raw/hermes --output-dir data/processed --distribution poisson --seed 42

# Validation mode - process only one OBJECTID
python src/preprocessing/main.py --input-dir data/raw/hermes --output-dir data/processed --distribution exact --objectid 1

# Full dataset with normal distribution
python src/preprocessing/main.py --input-dir data/raw/hermes --output-dir data/processed --distribution normal --seed 123
```

## 🚛 Hermes Trip Expansion

Atlas includes specialized functionality for processing Hermes transportation data:

### Trip Expansion Features
- **JSON Feature Processing**: Converts Hermes JSON features into individual trip records
- **2024 Date Generation**: Creates trip records for every day of 2024
- **Distribution Modes**: Three modes for generating realistic trip variability:
  - `exact`: Generate exactly the average number of trips per day
  - `poisson`: Use Poisson distribution around the average (realistic variability)
  - `normal`: Use truncated normal distribution around the average

### Distribution Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `exact` | Generates exactly the average trips per day | Testing, baseline analysis |
| `poisson` | Poisson distribution around average | Realistic daily variability |
| `normal` | Normal distribution around average | Alternative variability model |

### Validation Mode
Process specific routes for testing:
```bash
python src/preprocessing/main.py --objectid 1 --distribution exact
```

### File Naming Convention
Output files include distribution mode and scope:
- Single route: `processed_delivery_data_poisson_objectid_1.csv`
- Full dataset: `processed_delivery_data_exact_full_dataset.csv`

## 📊 Data Format

### Input Data
The preprocessor accepts multiple file formats:
- **CSV files** (`.csv`)
- **JSON files** (`.json`) - Including Hermes transportation data
- **Excel files** (`.xlsx`)
- **Text files** (`.txt`)

### Hermes JSON Format
For Hermes transportation data, the system processes JSON files with:
- **Features array**: Contains transportation route data
- **Attributes**: Route information including daily trip averages
- **Geometry**: Coordinate paths for routes
- **Trip expansion**: Converts weekly averages into daily trip records for 2024

Example Hermes feature structure:
```json
{
  "attributes": {
    "OBJECTID": 1,
    "nombre_zona_origen": "Vitoria-Gasteiz",
    "nombre_zona_destino": "Madrid",
    "viajes_OD_Lunes": 5,
    "viajes_OD_Martes": 6,
    "viajes_OD_Miercoles": 4,
    "viajes_OD_Jueves": 6,
    "viajes_OD_Viernes": 3,
    "viajes_OD_Sabado": 0,
    "viajes_OD_Domingo": 3
  }
}

### Expected Data Columns
The system can automatically map common column variations, but the following are the standard column names:

| Column | Description | Example |
|--------|-------------|---------|
| `delivery_id` | Unique identifier for delivery | `DEL_0001` |
| `pickup_latitude` | Pickup location latitude | `40.7128` |
| `pickup_longitude` | Pickup location longitude | `-74.0060` |
| `delivery_latitude` | Delivery location latitude | `40.7589` |
| `delivery_longitude` | Delivery location longitude | `-73.9851` |
| `pickup_address` | Pickup address | `123 Main St, NYC` |
| `delivery_address` | Delivery address | `456 Oak Ave, NYC` |
| `package_weight` | Package weight in kg | `15.5` |
| `package_volume` | Package volume in m³ | `0.25` |
| `delivery_deadline` | Delivery deadline | `2025-07-25` |
| `pickup_time_window_start` | Pickup window start | `2025-07-23 09:00:00` |
| `pickup_time_window_end` | Pickup window end | `2025-07-23 11:00:00` |
| `delivery_time_window_start` | Delivery window start | `2025-07-23 14:00:00` |
| `delivery_time_window_end` | Delivery window end | `2025-07-23 16:00:00` |
| `priority` | Delivery priority | `normal`, `high`, `urgent` |
| `package_type` | Type of package | `electronics`, `documents`, etc. |

### Output Data
The preprocessor outputs different formats depending on input:

#### Standard CSV Processing
For regular delivery data, outputs a standardized CSV file with:
- Cleaned and validated data
- Standardized column names
- Additional metadata columns:
  - `source_file`: Original file name
  - `processed_timestamp`: When the data was processed

#### Hermes Trip Expansion
For Hermes JSON data, outputs expanded trip records with:
- **Individual trip records**: One row per trip occurrence
- **Date information**: Full 2024 calendar with Spanish day names
- **Distribution metadata**: Mode used and actual vs average trips
- **Route details**: Origin/destination coordinates and zone information
- **Trip numbering**: Sequential numbering within each day

Example expanded output columns:
| Column | Description | Example |
|--------|-------------|---------|
| `objectid` | Route identifier | `1` |
| `fecha` | Trip date | `2024-01-15` |
| `dia_semana` | Day of week (Spanish) | `Lunes` |
| `nombre_zona_origen` | Origin zone name | `Vitoria-Gasteiz` |
| `nombre_zona_destino` | Destination zone name | `Madrid` |
| `numero_viaje_dia` | Trip number for that day | `1`, `2`, `3`... |
| `total_viajes_dia` | Total trips generated for day | `5` |
| `promedio_viajes_dia` | Original average for day | `5.0` |
| `modo_distribucion` | Distribution mode used | `poisson` |

## ⚙️ Configuration

The system uses a JSON configuration file (`config/config.json`) that controls:

- **Preprocessing settings**: File formats, column mappings, data validation rules
- **Data analysis parameters**: Coordinate precision, measurement units
- **Logging configuration**: Log levels, file locations

### Command Line Parameters

The preprocessing module supports various parameters:

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--input-dir` | Directory with raw data files | `data/raw/hermes` |
| `--output-dir` | Output directory for processed files | `data/processed` |
| `--distribution` | Distribution mode for trip generation | `exact`, `poisson`, `normal` |
| `--objectid` | Process specific OBJECTID (validation mode) | `1` |
| `--seed` | Random seed for reproducibility | `42` |
| `--config` | Custom configuration file path | `config/custom.json` |

Example configuration excerpt:
```json
{
  "preprocessing": {
    "coordinate_precision": 6,
    "weight_unit": "kg",
    "volume_unit": "m3",
    "output_filename": "processed_delivery_data.csv"
  }
}
```

## 🧪 Testing

### Basic Testing
Run the preprocessing pipeline:
```bash
python src/main.py preprocessing
```

### Hermes Data Testing
Test with specific distribution modes:

```bash
# Test exact distribution with validation
python src/preprocessing/main.py --input-dir data/raw/hermes --output-dir data/processed --distribution exact --objectid 1

# Test Poisson distribution with full dataset
python src/preprocessing/main.py --input-dir data/raw/hermes --output-dir data/processed --distribution poisson --seed 42

# Test normal distribution
python src/preprocessing/main.py --input-dir data/raw/hermes --output-dir data/processed --distribution normal --seed 123
```

This will:
1. Read your raw data files from `data/raw/`
2. Process the data through the preprocessing pipeline
3. Output a clean CSV file in `data/processed/` with appropriate naming
4. Display summary statistics including:
   - Total trip records generated
   - Distribution mode statistics
   - Date range covered
   - Unique routes processed

## 📁 Data Directories

- **`data/raw/`**: Place your raw data files here (CSV, JSON, Excel, etc.)
  - **`data/raw/hermes/`**: Specifically for Hermes transportation JSON files
- **`data/processed/`**: Processed CSV files will be saved here with descriptive names:
  - `processed_delivery_data_exact_objectid_1.csv` (single route validation)
  - `processed_delivery_data_poisson_full_dataset.csv` (complete dataset)
- **`data/interim/`**: For intermediate processing steps (future use)

## 🔍 Logging

All operations are logged with timestamps and details:
- Console output for immediate feedback
- Log files saved in `logs/` directory
- Configurable log levels (INFO, DEBUG, WARNING, ERROR)

## 🔧 Development

### Adding New Data Sources

1. Place raw data files in `data/raw/`
2. Update column mappings in the preprocessor if needed
3. Run preprocessing to validate the new data

### Extending the System

The modular structure allows easy extension:
- **New data formats**: Extend `FileHandler` in `src/utils/file_handler.py`
- **Enhanced analytics**: Add analysis modules for deeper insights
- **Data visualization**: Integrate visualization tools for data exploration

## 🤖 Model Evaluation & Comparison

Atlas includes tools for comparing trained machine learning models:

### Compare Models Tool

The `compare_models.py` script provides comprehensive model comparison with:
- **Visual tables**: Clean, tabulated output for easy reading
- **Per-task grouping**: Separate tables for probability, price, and weight models
- **Best model identification**: Automatic detection of best model per task (marked with ✓)
- **Size metrics**: Model size in MB for deployment considerations
- **Training metadata**: Date, features count, cross-validation folds
- **Multiple exports**: CSV and JSON formats for further analysis

#### Usage

```bash
# Compare all models
python scripts/compare_models.py

# Compare models for specific task
python scripts/compare_models.py --task probability

# Use custom models directory
python scripts/compare_models.py --models-dir path/to/models
```

#### Example Output

```
================================================================================
PROBABILITY MODELS
================================================================================

Model                                Date              Size MB    MAE         MAE_Std     R²          RMSE        Features
-----------------------------------  ----------------  ---------  ----------  ----------  ----------  ----------  ----------
probability_v20251025_190949         25/10/2025 19:09  28.4       0.005167    0.000309    0.998068    0.033553    19
probability_v20251025_220658         25/10/2025 22:06  28.4       0.005167    0.000309    0.998068    0.033553    19
───────────────────────────────────  ────────────────  ──────     ──────────  ──────────  ──────────  ──────────  ────────
✓ probability_v20251025_190949       25/10/2025 19:09  28.4       0.005167    0.000309    0.998068    0.033553    19
```

#### Output Files

Results are saved to `experiments/` directory:
- `model_comparison_YYYYMMDD_HHMMSS.csv`: All model metrics in CSV format
- `model_comparison_YYYYMMDD_HHMMSS.json`: Complete metrics in JSON format

## 📋 Future Roadmap

- [ ] Enhanced data analytics and visualization
- [ ] Machine learning for pattern recognition
- [ ] Real-time data processing capabilities
- [ ] Web-based dashboard for data exploration
- [ ] Integration with GIS systems
- [ ] Advanced distribution models for trip generation
- [ ] Route optimization analysis
- [ ] Temporal pattern analysis tools

## 🤝 Contributing

This is a post-graduate research project. Contributions and suggestions are welcome:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

[Add your license information here]

## 👥 Authors

Atlas Team - Post-graduate Research Project

## 📞 Support

For questions or issues, please [create an issue](../../issues) in the repository.

---

**Atlas** - Making truck delivery data analysis more accessible and insightful! 🚛✨
