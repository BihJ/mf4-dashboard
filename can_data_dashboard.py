#!/usr/bin/env python3

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
import glob
from datetime import datetime, timedelta
from asammdf import MDF
import canmatrix
from functools import lru_cache

def get_available_files():
    """Get list of available MF4U and CSV files"""
    # Get files from D1FX folder
    mf4u_files = glob.glob("!D1FX/*.mf4u")
    
    # Get uploaded MF4U files
    uploaded_mf4u_files = glob.glob("uploaded_mf4u_*.mf4u")
    
    # Get CSV files from output folder
    csv_files = glob.glob("output/*_decoded.csv")
    
    # Combine all MF4U files
    all_mf4u_files = sorted(mf4u_files + uploaded_mf4u_files)
    
    return {
        'mf4u_files': all_mf4u_files,
        'csv_files': sorted(csv_files)
    }

def convert_mf4u_to_csv(input_file, dbc_file="NETWORK-configuration/MainCanNetwork.dbc"):
    """Convert MF4U file to decoded CSV"""
    try:
        filename = os.path.basename(input_file).replace('.mf4u', '')
        raw_csv = os.path.join('output', f'{filename}_raw.csv')
        decoded_csv = os.path.join('output', f'{filename}_decoded.csv')

        # Create output directory if it doesn't exist
        os.makedirs('output', exist_ok=True)

        # Convert to raw CSV
        mdf = MDF(input_file)
        df_raw = mdf.to_dataframe()
        df_raw.to_csv(raw_csv, index=True)

        # Decode with DBC - try both MainCanNetwork and InverterCanNetwork
        main_dbc = "NETWORK-configuration/MainCanNetwork.dbc"
        inverter_dbc = "NETWORK-configuration/InverterCanNetwork.dbc"

        # Try with both DBC files to get complete signal coverage
        databases = {"CAN": [(main_dbc, 0), (inverter_dbc, 0)]}
        decoded_mdf = mdf.extract_bus_logging(database_files=databases)
        df_decoded = decoded_mdf.to_dataframe(use_display_names=True)
        df_decoded.to_csv(decoded_csv, index=True)

        return True, f"Successfully converted {input_file}"
    except Exception as e:
        return False, f"Error converting {input_file}: {str(e)}"

def convert_uploaded_mf4u_to_csv(input_file, uploaded_dbc_files=None):
    """Convert uploaded MF4U file to decoded CSV using uploaded DBC files"""
    try:
        filename = os.path.basename(input_file).replace('.mf4u', '').replace('uploaded_mf4u_', '')
        raw_csv = os.path.join('output', f'{filename}_raw.csv')
        decoded_csv = os.path.join('output', f'{filename}_decoded.csv')

        # Create output directory if it doesn't exist
        os.makedirs('output', exist_ok=True)

        # Convert to raw CSV
        mdf = MDF(input_file)
        df_raw = mdf.to_dataframe()
        df_raw.to_csv(raw_csv, index=True)

        # Prepare DBC files for decoding
        databases = {"CAN": []}
        
        # Add uploaded DBC files
        if uploaded_dbc_files:
            for dbc_file in uploaded_dbc_files:
                dbc_path = f"uploaded_dbc_{dbc_file.name}"
                if os.path.exists(dbc_path):
                    databases["CAN"].append((dbc_path, 0))
        
        # Add default DBC files if they exist
        main_dbc = "NETWORK-configuration/MainCanNetwork.dbc"
        inverter_dbc = "NETWORK-configuration/InverterCanNetwork.dbc"
        
        if os.path.exists(main_dbc):
            databases["CAN"].append((main_dbc, 0))
        if os.path.exists(inverter_dbc):
            databases["CAN"].append((inverter_dbc, 0))
        
        # If no DBC files available, return raw data
        if not databases["CAN"]:
            df_raw.to_csv(decoded_csv, index=True)
            return True, f"Successfully converted {filename} (raw data only - no DBC files)"
        
        # Decode with available DBC files
        decoded_mdf = mdf.extract_bus_logging(database_files=databases)
        df_decoded = decoded_mdf.to_dataframe(use_display_names=True)
        df_decoded.to_csv(decoded_csv, index=True)

        return True, f"Successfully converted {filename} with {len(databases['CAN'])} DBC files"
    except Exception as e:
        return False, f"Error converting {input_file}: {str(e)}"

def load_data(selected_file=None):
    """Load the decoded CAN data from selected file"""
    if selected_file is None:
        # Default to first available file
        csv_files = glob.glob("output/*_decoded.csv")
        if csv_files:
            selected_file = csv_files[0]
        else:
            st.error("No decoded CSV files found. Please convert some MF4U files first.")
            return None
    
    if os.path.exists(selected_file):
        try:
            df = pd.read_csv(selected_file)
            df['timestamps'] = pd.to_numeric(df['timestamps'], errors='coerce')
            return df
        except Exception as e:
            st.error(f"Error loading file {selected_file}: {str(e)}")
            return None
    else:
        st.error(f"CSV file not found: {selected_file}")
        return None

def parse_dbc_file(dbc_file_path):
    """Parse DBC file and extract message and signal information"""
    try:
        # Load DBC file - returns a dictionary
        db_dict = canmatrix.formats.loadp(dbc_file_path)
        
        # Get the CanMatrix object (usually the first/only value)
        db = list(db_dict.values())[0]
        
        # Create mapping of signals to their message names and descriptions
        signal_info = {}
        message_categories = {}
        
        for frame in db.frames:
            frame_name = frame.name
            frame_comment = frame.comment if hasattr(frame, 'comment') and frame.comment else ""
            
            # Create category based on frame name patterns
            category = categorize_message(frame_name, frame_comment)
            
            if category not in message_categories:
                message_categories[category] = []
            
            message_categories[category].append({
                'name': frame_name,
                'comment': frame_comment,
                'signals': []
            })
            
            # Add signals for this frame
            for signal in frame.signals:
                signal_name = signal.name
                signal_comment = signal.comment if hasattr(signal, 'comment') and signal.comment else ""
                
                # Create a more descriptive signal name if there are duplicates
                if signal_name in signal_info:
                    # If signal name already exists, create a unique name with frame context
                    unique_signal_name = f"{frame_name}_{signal_name}"
                else:
                    unique_signal_name = signal_name
                
                signal_info[unique_signal_name] = {
                    'message': frame_name,
                    'category': category,
                    'comment': signal_comment,
                    'original_name': signal_name
                }
                
                message_categories[category][-1]['signals'].append({
                    'name': signal_name,
                    'comment': signal_comment
                })
        
        return signal_info, message_categories
    except Exception as e:
        st.error(f"Error parsing DBC file: {e}")
        return {}, {}

def categorize_message(message_name, message_comment):
    """Categorize message based purely on DBC file structure - no hardcoded patterns"""
    # Simply use the message name as the category
    # This ensures all signals are properly categorized without any filtering
    return message_name


@lru_cache(maxsize=1)
def create_signal_name_mapping_cached(
    columns_tuple, dbc_file_path="NETWORK-configuration/MainCanNetwork.dbc"
):
    """Create a mapping from display names to shorter readable names"""
    try:
        name_mapping = {}

        # Get all columns except timestamps
        signal_columns = [col for col in columns_tuple if col != "timestamps"]

        for col in signal_columns:
            # Parse display name format: CAN2.Wheel_1_Setpoints.Torque
            if '.' in col:
                parts = col.split(".", 2)  # Split into max 3 parts for efficiency
                if len(parts) >= 3:
                    # Create shorter name: Wheel_1_Setpoints_Torque
                    frame_name = parts[1]
                    signal_name = parts[2]
                    readable_name = f"{frame_name}_{signal_name}"
                    name_mapping[col] = readable_name
                else:
                    name_mapping[col] = col
            else:
                # For signals without display name format, keep original
                name_mapping[col] = col

        return name_mapping
    except Exception as e:
        # If there's an error, return identity mapping
        return {col: col for col in columns_tuple}


def create_signal_name_mapping(
    df, dbc_file_path="NETWORK-configuration/MainCanNetwork.dbc"
):
    """Wrapper function for cached signal name mapping"""
    return create_signal_name_mapping_cached(tuple(df.columns), dbc_file_path)


@lru_cache(maxsize=1)
def get_signal_categories_cached(
    columns_tuple, dbc_file_path="NETWORK-configuration/MainCanNetwork.dbc"
):
    """Categorize signals based on display names (e.g., CAN2.Wheel_1_Setpoints.Torque)"""
    try:
        # Create categories based on signal display names - optimized version
        categories = {}

        # Get all columns except timestamps
        signal_columns = [col for col in columns_tuple if col != "timestamps"]

        for col in signal_columns:
            # Parse display name format: CAN2.Wheel_1_Setpoints.Torque
            if '.' in col:
                parts = col.split(".", 2)  # Split into max 3 parts for efficiency
                if len(parts) >= 3:
                    # Extract frame name (e.g., Wheel_1_Setpoints from CAN2.Wheel_1_Setpoints.Torque)
                    frame_name = parts[1]
                    categories.setdefault(frame_name, []).append(col)
                else:
                    # Fallback for unexpected format
                    categories.setdefault("Other", []).append(col)
            else:
                # For signals without display name format, put in Other
                categories.setdefault("Other", []).append(col)

        # Remove empty categories
        categories = {k: v for k, v in categories.items() if v}

        return categories
    except Exception as e:
        # Fallback: create a single category with all signals
        return {"All Signals": [col for col in columns_tuple if col != "timestamps"]}


def get_signal_categories(df, dbc_file_path="NETWORK-configuration/MainCanNetwork.dbc"):
    """Wrapper function for cached signal categories"""
    return get_signal_categories_cached(tuple(df.columns), dbc_file_path)


def create_time_series_plot(df, selected_signals, time_range=None):
    """Create time series plot for selected signals"""
    if not selected_signals:
        return None

    fig = go.Figure()

    # Limit data points for performance (max 10,000 points per signal)
    max_points = 10000

    for signal in selected_signals:
        if signal in df.columns:
            data = df[['timestamps', signal]].dropna()
            if time_range:
                data = data[(data['timestamps'] >= time_range[0]) & (data['timestamps'] <= time_range[1])]

            # Sample data if too many points
            if len(data) > max_points:
                step = len(data) // max_points
                data = data.iloc[::step]

            fig.add_trace(go.Scatter(
                x=data['timestamps'],
                y=data[signal],
                mode='lines',
                name=signal,
                line=dict(width=1)
            ))

    fig.update_layout(
        title="Time Series Plot",
        xaxis_title="Time (seconds)",
        yaxis_title="Value",
        hovermode="x unified",
    )

    return fig

def create_correlation_heatmap(df, selected_signals):
    """Create correlation heatmap for selected signals"""
    if len(selected_signals) < 2:
        return None
    
    # Filter to only numeric columns
    numeric_signals = [s for s in selected_signals if s in df.columns and pd.api.types.is_numeric_dtype(df[s])]
    
    if len(numeric_signals) < 2:
        return None
    
    corr_data = df[numeric_signals].corr()
    
    fig = px.imshow(
        corr_data,
        text_auto=True,
        aspect="auto",
        title="Signal Correlation Matrix"
    )
    
    fig.update_layout(height=600)
    return fig

def create_histogram_plot(df, selected_signals):
    """Create histogram for selected signals"""
    if not selected_signals:
        return None
    
    # Filter to only numeric columns
    numeric_signals = [s for s in selected_signals if s in df.columns and pd.api.types.is_numeric_dtype(df[s])]
    
    if not numeric_signals:
        return None
    
    fig = go.Figure()
    
    for signal in numeric_signals[:5]:  # Limit to 5 signals for readability
        fig.add_trace(go.Histogram(
            x=df[signal].dropna(),
            name=signal,
            opacity=0.7
        ))
    
    fig.update_layout(
        title="Signal Distribution",
        xaxis_title="Value",
        yaxis_title="Frequency",
        barmode='overlay',
        height=500
    )
    
    return fig

def create_scatter_plot(df, x_signal, y_signal):
    """Create scatter plot between two signals"""
    if x_signal not in df.columns or y_signal not in df.columns:
        return None
    
    fig = px.scatter(
        df,
        x=x_signal,
        y=y_signal,
        title=f"{y_signal} vs {x_signal}",
        trendline="ols"
    )
    
    fig.update_layout(height=500)
    return fig

def main():
    st.set_page_config(
        page_title="CAN Bus Data Dashboard",
        page_icon="🚗",
        layout="wide"
    )

    st.title("🚗 CAN Bus Data Dashboard")
    st.markdown("Interactive analysis of vehicle CAN bus data")

    # File management section
    st.sidebar.header("📁 File Management")
    
    # File upload section
    with st.sidebar.expander("📤 Upload Files", expanded=False):
        st.write("**Upload DBC Files:**")
        uploaded_dbc_files = st.file_uploader(
            "Choose DBC files",
            type=['dbc'],
            accept_multiple_files=True,
            help="Upload CAN database files (.dbc) for decoding"
        )
        
        st.write("**Upload MF4U Files:**")
        uploaded_mf4u_files = st.file_uploader(
            "Choose MF4U files", 
            type=['mf4u'],
            accept_multiple_files=True,
            help="Upload CAN bus log files (.mf4u) for conversion"
        )
        
        # Process uploaded files
        if uploaded_dbc_files:
            for uploaded_file in uploaded_dbc_files:
                # Save uploaded DBC file
                dbc_path = f"uploaded_dbc_{uploaded_file.name}"
                with open(dbc_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"✅ Uploaded: {uploaded_file.name}")
        
        if uploaded_mf4u_files:
            for uploaded_file in uploaded_mf4u_files:
                # Save uploaded MF4U file
                mf4u_path = f"uploaded_mf4u_{uploaded_file.name}"
                with open(mf4u_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"✅ Uploaded: {uploaded_file.name}")
                
                # Auto-convert uploaded MF4U file
                if st.button(f"Convert {uploaded_file.name}", key=f"convert_{uploaded_file.name}"):
                    with st.spinner(f"Converting {uploaded_file.name}..."):
                        success, message = convert_uploaded_mf4u_to_csv(mf4u_path, uploaded_dbc_files)
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
                        st.rerun()
        
        # Show uploaded files and cleanup options
        uploaded_dbc_files_existing = glob.glob("uploaded_dbc_*.dbc")
        uploaded_mf4u_files_existing = glob.glob("uploaded_mf4u_*.mf4u")
        
        if uploaded_dbc_files_existing or uploaded_mf4u_files_existing:
            st.write("**Uploaded Files:**")
            
            if uploaded_dbc_files_existing:
                st.write("DBC files:")
                for dbc_file in uploaded_dbc_files_existing:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"• {os.path.basename(dbc_file).replace('uploaded_dbc_', '')}")
                    with col2:
                        if st.button("🗑️", key=f"delete_dbc_{dbc_file}", help="Delete file"):
                            os.remove(dbc_file)
                            st.rerun()
            
            if uploaded_mf4u_files_existing:
                st.write("MF4U files:")
                for mf4u_file in uploaded_mf4u_files_existing:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"• {os.path.basename(mf4u_file).replace('uploaded_mf4u_', '')}")
                    with col2:
                        if st.button("🗑️", key=f"delete_mf4u_{mf4u_file}", help="Delete file"):
                            os.remove(mf4u_file)
                            st.rerun()
            
            # Clear all uploaded files button
            if st.button("🗑️ Clear All Uploaded Files", key="clear_all_uploaded"):
                for file in uploaded_dbc_files_existing + uploaded_mf4u_files_existing:
                    try:
                        os.remove(file)
                    except:
                        pass
                st.rerun()

    # Get available files (including uploaded ones)
    files = get_available_files()

    # File conversion section
    with st.sidebar.expander("🔄 Convert MF4U Files", expanded=False):
        st.write("**Available MF4U Files:**")
        if files['mf4u_files']:
            for mf4u_file in files['mf4u_files']:
                filename = os.path.basename(mf4u_file)
                st.write(f"• {filename}")

            st.write("**Convert Files:**")

            # Single file conversion
            selected_mf4u = st.selectbox(
                "Select MF4U file to convert:",
                options=files['mf4u_files'],
                format_func=lambda x: os.path.basename(x),
                key="single_convert"
            )

            if st.button("Convert Selected File", key="convert_single"):
                with st.spinner("Converting file..."):
                    success, message = convert_mf4u_to_csv(selected_mf4u)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)

            # Batch conversion
            if st.button("Convert All Files", key="convert_all"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                total_files = len(files['mf4u_files'])
                success_count = 0

                for i, mf4u_file in enumerate(files['mf4u_files']):
                    filename = os.path.basename(mf4u_file)
                    status_text.text(f"Converting {filename}...")

                    success, message = convert_mf4u_to_csv(mf4u_file)
                    if success:
                        success_count += 1

                    progress_bar.progress((i + 1) / total_files)

                status_text.text(f"Conversion complete! {success_count}/{total_files} files converted successfully.")
                st.rerun()
        else:
            st.info("No MF4U files found in !D1FX/ folder")

    # File selection section
    st.sidebar.subheader("📊 Select Data File")

    # Refresh files list
    files = get_available_files()

    if files['csv_files']:
        selected_file = st.sidebar.selectbox(
            "Choose decoded CSV file:",
            options=files['csv_files'],
            format_func=lambda x: os.path.basename(x),
            key="file_selector"
        )

        # Show file info
        file_size = os.path.getsize(selected_file) / (1024 * 1024)  # MB
        st.sidebar.info(f"File: {os.path.basename(selected_file)}\nSize: {file_size:.1f} MB")
    else:
        st.sidebar.warning("No decoded CSV files found. Convert some MF4U files first.")
        selected_file = None

    # Load data
    with st.spinner("Loading data..."):
        df = load_data(selected_file)

    if df is None:
        st.stop()

    # Sidebar filters
    st.sidebar.header("📊 Filters & Controls")

    # Time range filter
    time_min = df['timestamps'].min()
    time_max = df['timestamps'].max()

    st.sidebar.subheader("⏱️ Time Range")
    time_range = st.sidebar.slider(
        "Select time range (seconds)",
        min_value=float(time_min),
        max_value=float(time_max),
        value=(float(time_min), float(time_max)),
        step=0.1
    )

    # Filter data by time range
    df_filtered = df[(df['timestamps'] >= time_range[0]) & (df['timestamps'] <= time_range[1])]

    # Signal categories based on actual decoded signals (from both MainCanNetwork and InverterCanNetwork)
    main_dbc = "NETWORK-configuration/MainCanNetwork.dbc"
    inverter_dbc = "NETWORK-configuration/InverterCanNetwork.dbc"
    categories = get_signal_categories(
        df
    )  # Use actual decoded signals, not DBC parsing
    name_mapping = create_signal_name_mapping(
        df
    )  # Use actual decoded signals, not DBC parsing

    # Display DBC file information
    with st.sidebar.expander("📋 DBC File Info", expanded=False):
        # Show DBC files used
        dbc_files_used = []
        if os.path.exists(main_dbc):
            dbc_files_used.append(os.path.basename(main_dbc))
        if os.path.exists(inverter_dbc):
            dbc_files_used.append(os.path.basename(inverter_dbc))
        
        # Check for uploaded DBC files
        uploaded_dbc_files = glob.glob("uploaded_dbc_*.dbc")
        for dbc_file in uploaded_dbc_files:
            dbc_files_used.append(os.path.basename(dbc_file).replace("uploaded_dbc_", ""))
        
        if dbc_files_used:
            st.write(f"**DBC Files Used:** {', '.join(dbc_files_used)}")
        else:
            st.write("**DBC Files Used:** None (raw data only)")
            
        st.write(f"**Decoded Signals:** {len(df.columns) - 1} total signals")
        st.write(f"**Signal Categories:** {len(categories)} categories")

        st.write("**Available Categories:**")
        for category_name, signals in list(categories.items())[
            :10
        ]:  # Show first 10 categories
            st.write(f"• {category_name}: {len(signals)} signals")
        if len(categories) > 10:
            st.write(f"... and {len(categories) - 10} more categories")

    st.sidebar.subheader("📡 Signal Categories")
    selected_categories = st.sidebar.multiselect(
        "Select signal categories",
        options=list(categories.keys()),
        default=list(categories.keys())[:3]  # Select first 3 by default
    )

    # Get signals from selected categories
    selected_signals = []
    for category in selected_categories:
        selected_signals.extend(categories[category])

    # Remove timestamps from signal list
    selected_signals = [s for s in selected_signals if s != "timestamps"]

    # Additional signal selection
    st.sidebar.subheader("🎯 Specific Signals")

    # Create readable signal options
    signal_options = []
    for signal in selected_signals:
        readable_name = name_mapping.get(signal, signal)
        signal_options.append((readable_name, signal))

    # Set default signals if none are selected
    default_signals = selected_signals[:5] if selected_signals else []

    specific_signals = st.sidebar.multiselect(
        "Select specific signals",
        options=[opt[1] for opt in signal_options],
        format_func=lambda x: name_mapping.get(x, x),
        default=default_signals,
        key="specific_signals_selector",
    )

    # If no signals are selected, use the default signals
    if not specific_signals and default_signals:
        specific_signals = default_signals

    # Final fallback: if still no signals, use first few from selected_signals
    if not specific_signals and selected_signals:
        specific_signals = selected_signals[:3]

    # Main content
    st.subheader(f"📊 Analyzing: {os.path.basename(selected_file) if selected_file else 'No file selected'}")

    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

    with col1:
        st.metric("Total Data Points", f"{len(df_filtered):,}")
    with col2:
        st.metric("Time Duration", f"{time_range[1] - time_range[0]:.1f}s")
    with col3:
        st.metric("Selected Signals", len(specific_signals))
    with col4:
        st.metric("Total Signals", len(df.columns) - 1)

    # File comparison section
    if len(files['csv_files']) > 1:
        with st.expander("📈 Compare Files", expanded=False):
            st.write("**File Comparison:**")

            # Create comparison table
            comparison_data = []
            for csv_file in files['csv_files']:
                try:
                    temp_df = pd.read_csv(csv_file)
                    file_size = os.path.getsize(csv_file) / (1024 * 1024)  # MB
                    time_duration = temp_df['timestamps'].max() - temp_df['timestamps'].min()

                    comparison_data.append({
                        'File': os.path.basename(csv_file),
                        'Data Points': len(temp_df),
                        'Signals': len(temp_df.columns) - 1,
                        'Duration (s)': f"{time_duration:.1f}",
                        'Size (MB)': f"{file_size:.1f}",
                        'Start Time': f"{temp_df['timestamps'].min():.2f}",
                        'End Time': f"{temp_df['timestamps'].max():.2f}"
                    })
                except Exception as e:
                    comparison_data.append({
                        'File': os.path.basename(csv_file),
                        'Data Points': 'Error',
                        'Signals': 'Error',
                        'Duration (s)': 'Error',
                        'Size (MB)': 'Error',
                        'Start Time': 'Error',
                        'End Time': 'Error'
                    })

            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, width="stretch")

    # Tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 Time Series", "🔥 Correlation", "📊 Distribution", "🎯 Scatter Plot", "📋 Data Table", "🔄 Multi-File Analysis"])

    with tab1:
        st.subheader("Time Series Analysis")
        if specific_signals:
            fig_time = create_time_series_plot(df_filtered, specific_signals, time_range)
            if fig_time:
                st.plotly_chart(fig_time, width="stretch")
            else:
                st.warning("No valid signals selected for time series plot")
        else:
            st.info("Please select signals to plot")

    with tab2:
        st.subheader("Signal Correlation Analysis")
        if len(specific_signals) >= 2:
            fig_corr = create_correlation_heatmap(df_filtered, specific_signals)
            if fig_corr:
                st.plotly_chart(fig_corr, width="stretch")
            else:
                st.warning("Need at least 2 numeric signals for correlation analysis")
        else:
            st.info("Please select at least 2 signals for correlation analysis")

    with tab3:
        st.subheader("Signal Distribution")
        if specific_signals:
            fig_hist = create_histogram_plot(df_filtered, specific_signals)
            if fig_hist:
                st.plotly_chart(fig_hist, width="stretch")
            else:
                st.warning("No numeric signals selected for histogram")
        else:
            st.info("Please select signals for distribution analysis")

    with tab4:
        st.subheader("Scatter Plot Analysis")
        col_x, col_y = st.columns(2)

        with col_x:
            x_signal = st.selectbox(
                "X-axis signal", 
                options=specific_signals, 
                format_func=lambda x: name_mapping.get(x, x),
                key="x_signal"
            )
        with col_y:
            y_signal = st.selectbox(
                "Y-axis signal", 
                options=specific_signals, 
                format_func=lambda x: name_mapping.get(x, x),
                key="y_signal"
            )

        if x_signal and y_signal and x_signal != y_signal:
            fig_scatter = create_scatter_plot(df_filtered, x_signal, y_signal)
            if fig_scatter:
                st.plotly_chart(fig_scatter, width="stretch")
        else:
            st.info("Please select two different signals for scatter plot")

    with tab5:
        st.subheader("Data Table")

        # Show basic statistics
        if specific_signals:
            st.write("**Selected Signals Statistics:**")
            numeric_signals = [s for s in specific_signals if s in df_filtered.columns and pd.api.types.is_numeric_dtype(df_filtered[s])]

            if numeric_signals:
                stats_df = df_filtered[numeric_signals].describe()
                st.dataframe(stats_df, width="stretch")

            # Show sample data
            st.write("**Sample Data:**")
            display_df = df_filtered[['timestamps'] + specific_signals[:10]]  # Limit to 10 columns for display
            st.dataframe(display_df.head(100), width="stretch")
        else:
            st.info("Please select signals to view data table")

    with tab6:
        st.subheader("Multi-File Analysis")

        if len(files['csv_files']) > 1:
            st.write("**Compare signals across multiple files:**")

            # File selection for comparison
            col_file1, col_file2 = st.columns(2)

            with col_file1:
                file1 = st.selectbox(
                    "Select first file:",
                    options=files['csv_files'],
                    format_func=lambda x: os.path.basename(x),
                    key="multi_file1"
                )

            with col_file2:
                file2 = st.selectbox(
                    "Select second file:",
                    options=files['csv_files'],
                    format_func=lambda x: os.path.basename(x),
                    key="multi_file2"
                )

            if file1 != file2:
                # Load both files
                try:
                    df1 = pd.read_csv(file1)
                    df2 = pd.read_csv(file2)

                    # Find common signals
                    common_signals = list(set(df1.columns) & set(df2.columns))
                    common_signals = [s for s in common_signals if s != 'timestamps']

                    if common_signals:
                        # Signal selection for comparison
                        compare_signal = st.selectbox(
                            "Select signal to compare:",
                            options=common_signals,
                            format_func=lambda x: name_mapping.get(x, x),
                            key="multi_signal"
                        )

                        if compare_signal:
                            # Create comparison plot
                            fig = go.Figure()

                            # Add data from file 1
                            fig.add_trace(go.Scatter(
                                x=df1['timestamps'],
                                y=df1[compare_signal],
                                mode='lines',
                                name=f"{os.path.basename(file1)}",
                                line=dict(width=1)
                            ))

                            # Add data from file 2
                            fig.add_trace(go.Scatter(
                                x=df2['timestamps'],
                                y=df2[compare_signal],
                                mode='lines',
                                name=f"{os.path.basename(file2)}",
                                line=dict(width=1)
                            ))

                            fig.update_layout(
                                title=f"Signal Comparison: {compare_signal}",
                                xaxis_title="Time (seconds)",
                                yaxis_title="Value",
                                hovermode="x unified",
                            )

                            st.plotly_chart(fig, width="stretch")

                            # Statistics comparison
                            col_stat1, col_stat2 = st.columns(2)

                            with col_stat1:
                                st.write(f"**{os.path.basename(file1)} Statistics:**")
                                stats1 = df1[compare_signal].describe()
                                st.dataframe(stats1, width="stretch")

                            with col_stat2:
                                st.write(f"**{os.path.basename(file2)} Statistics:**")
                                stats2 = df2[compare_signal].describe()
                                st.dataframe(stats2, width="stretch")
                    else:
                        st.warning("No common signals found between the selected files.")

                except Exception as e:
                    st.error(f"Error loading files for comparison: {str(e)}")
            else:
                st.info("Please select two different files for comparison.")
        else:
            st.info("Need at least 2 decoded CSV files to perform multi-file analysis.")

    # Footer
    st.markdown("---")
    st.markdown("**CAN Bus Data Dashboard** - Built with Streamlit and Plotly")

if __name__ == "__main__":
    main()
