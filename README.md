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
│   │   └── main.py              # Preprocessing entry point
│   ├── utils/                    # Utility functions
│   │   ├── __init__.py
│   │   ├── logger.py            # Logging utilities
│   │   └── file_handler.py      # File handling utilities
│   ├── __init__.py
│   └── main.py                  # Main project entry point
├── data/                         # Data directories
│   ├── raw/                     # Raw input data files
│   ├── processed/               # Processed CSV outputs
│   └── interim/                 # Intermediate processing files
├── config/                       # Configuration files
│   └── config.json              # Main configuration
├── notebooks/                    # Jupyter notebooks for analysis
├── tests/                        # Unit tests
├── docs/                         # Documentation
├── logs/                         # Log files (auto-generated)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

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
python src/preprocessing/main.py --input-dir data/raw --output-dir data/processed --config config/config.json
```

## 📊 Data Format

### Input Data
The preprocessor accepts multiple file formats:
- **CSV files** (`.csv`)
- **JSON files** (`.json`)
- **Excel files** (`.xlsx`)
- **Text files** (`.txt`)

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
The preprocessor outputs a standardized CSV file with:
- Cleaned and validated data
- Standardized column names
- Additional metadata columns:
  - `source_file`: Original file name
  - `processed_timestamp`: When the data was processed

## ⚙️ Configuration

The system uses a JSON configuration file (`config/config.json`) that controls:

- **Preprocessing settings**: File formats, column mappings, data validation rules
- **Data analysis parameters**: Coordinate precision, measurement units
- **Logging configuration**: Log levels, file locations

Example configuration excerpt:
```json
{
  "preprocessing": {
    "coordinate_precision": 6,
    "weight_unit": "kg",
    "volume_unit": "m3"
  }
}
```

## 🧪 Testing

Run the preprocessing pipeline:
```bash
python src/main.py preprocessing
```

This will:
1. Read your raw data files from `data/raw/`
2. Process the data through the preprocessing pipeline
3. Output a clean CSV file in `data/processed/`
4. Display summary statistics

## 📁 Data Directories

- **`data/raw/`**: Place your raw data files here (CSV, JSON, Excel, etc.)
- **`data/processed/`**: Processed CSV files will be saved here
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

## 📋 Future Roadmap

- [ ] Enhanced data analytics and visualization
- [ ] Machine learning for pattern recognition
- [ ] Real-time data processing capabilities
- [ ] Web-based dashboard for data exploration
- [ ] Integration with GIS systems

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
