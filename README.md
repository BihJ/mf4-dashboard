# CAN Bus Data Converter & Dashboard

This project provides tools to convert Vector MF4U CAN bus log files to CSV format and analyze them with an interactive dashboard.

## Files Overview

### Core Scripts
- `can_data_dashboard.py` - Interactive Streamlit dashboard with file conversion and data analysis
- `run_dashboard.py` - Launcher script for the dashboard

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch Interactive Dashboard
```bash
# Option 1: Use launcher script
python run_dashboard.py

# Option 2: Direct streamlit command
streamlit run can_data_dashboard.py
```

### 3. Convert and Analyze Files
1. Open the dashboard in your browser
2. Use the "🔄 Convert MF4U Files" section in the sidebar
3. Select individual files or convert all at once
4. Switch between converted files using the file selector
5. Compare files using the "📈 Compare Files" section

## Dashboard Features

The interactive dashboard provides:

### 📊 **Data Visualization**
- **Time Series Plots** - Plot multiple signals over time
- **Correlation Heatmaps** - Find relationships between signals
- **Distribution Histograms** - Analyze signal distributions
- **Scatter Plots** - Compare two signals with trend lines
- **Multi-File Analysis** - Compare signals across different files

### 🎛️ **Interactive Controls**
- **Time Range Filter** - Focus on specific time periods
- **Signal Categories** - Filter by vehicle system (BMS, Motor, etc.)
- **Specific Signal Selection** - Choose individual signals to analyze
- **File Selection** - Switch between different converted files

### 📋 **Data Analysis**
- **Statistics Table** - View descriptive statistics
- **Data Table** - Browse raw data
- **Real-time Filtering** - Apply filters and see results instantly
- **File Comparison** - Compare file sizes, durations, and data points

### 🔄 **File Management**
- **MF4U to CSV Conversion** - Convert files directly in the dashboard
- **Batch Conversion** - Convert all MF4U files at once
- **File Browser** - View all available files and their properties
- **Progress Tracking** - Monitor conversion progress with progress bars

### 🚗 **Signal Categories**
- **Battery Management System (BMS)** - Cell voltages, temperatures, SOC
- **Motor/Inverter** - Motor temperatures, velocities, status
- **Power Measurement Board (PMB)** - Battery voltages, currents
- **Dashboard Controls** - Button states, user inputs
- **Wheel Control** - Torque and speed setpoints
- **Pedal Box** - Steering, brake, throttle inputs
- **IMU Sensors** - Accelerometer, gyroscope data
- **Charging System** - On-board charger status
- **Cooling System** - Pump controls, flow meters

## File Structure

```
dbc_converter/
├── !D1FX/                         # Input MF4U files
├── NETWORK-configuration/         # DBC files
├── output/                        # Generated CSV files
│   ├── *_raw.csv                 # Raw CAN data
│   └── *_decoded.csv             # Decoded signals
├── can_data_dashboard.py         # Streamlit dashboard (all-in-one)
├── run_dashboard.py              # Dashboard launcher
├── requirements.txt              # Dependencies
└── README.md                     # Documentation
```

## Usage Examples

### Launch Dashboard
```bash
python run_dashboard.py
# Then open http://localhost:8501 in your browser
```

### Convert Files in Dashboard
1. Open the dashboard
2. Use the "🔄 Convert MF4U Files" section
3. Select files to convert or convert all at once
4. Analyze the converted data using the various tabs

## Data Analysis Tips

1. **Start with Time Series** - Get an overview of signal behavior over time
2. **Use Correlation Analysis** - Find related signals (e.g., battery voltage vs current)
3. **Filter by Time Range** - Focus on specific driving events or conditions
4. **Compare Signal Categories** - Analyze different vehicle systems together
5. **Check Distributions** - Identify normal vs abnormal signal ranges

## Troubleshooting

### Dashboard Won't Start
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check if port 8501 is available
- Verify the decoded CSV file exists in the `output/` folder

### No Data in Dashboard
- Run the decoder first: `python decode_can_messages.py`
- Check that `output/D1F60_decoded.csv` exists and has data

### Performance Issues
- Use time range filters to reduce data size
- Select fewer signals for analysis
- The dashboard loads all data into memory - large files may be slow

## Technical Details

- **Input Format**: Vector MF4U files (CAN bus logs)
- **DBC Support**: Full DBC file parsing and signal decoding
- **Output Format**: CSV with decoded signals and timestamps
- **Dashboard**: Streamlit + Plotly for interactive visualization
- **Data Processing**: asammdf library for MF4U handling
