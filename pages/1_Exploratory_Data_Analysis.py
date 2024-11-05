import os
import pandas as pd
import streamlit as st
import numpy as np
from scipy import fft
import plotly.graph_objects as go

from data_extraction import bucket_name, credentials
from data_extraction import check_and_download

folder_path = "./data/dashboard"


# Utility functions to load data
def load_csv_data(file_path, delimiter=","):
    try:
        return pd.read_csv(file_path, delimiter=delimiter)
    except Exception as e:
        st.error(f"Error loading data: {e} (Tried path: {file_path})")
        return None


def load_wear_data(selected_chart):
    wear_file_path = os.path.join(
        folder_path, selected_chart.lower(), f"{selected_chart.lower()}_wear.csv"
    )
    return load_csv_data(wear_file_path)


def load_stats_data(selected_chart):
    stats_file_path = os.path.join(
        folder_path, selected_chart.lower(), f"{selected_chart.lower()}_statistics.csv"
    )
    return load_csv_data(stats_file_path)


def load_main_data(selected_chart):
    data_folder = os.path.join(
        folder_path, selected_chart.lower(), selected_chart.lower()
    )
    csv_files = [
        f for f in os.listdir(data_folder) if f.endswith(".csv") and "_freq" not in f
    ]
    if not csv_files:
        st.error("No valid data files found.")
        return None
    return load_csv_data(os.path.join(data_folder, csv_files[0]))


st.set_page_config(page_title="Exploratory Data Analysis", layout="wide")

st.title("Exploratory Data Analysis")

col_dataset, col_cut_file, col_signal = st.columns(3)

# Dataset Selection
with col_dataset:
    selected_chart = st.radio(
        "Select Dataset", ["C1", "C4", "C6"], key="chart_selection", horizontal=True
    )

# Check and download wear and statistics files for the selected dataset
wear_file_path = f"dashboard/{selected_chart.lower()}/{selected_chart.lower()}_wear.csv"
statistics_file_path = (
    f"dashboard/{selected_chart.lower()}/{selected_chart.lower()}_statistics.csv"
)

check_and_download(bucket_name, wear_file_path, "./data/" + wear_file_path, credentials)
check_and_download(
    bucket_name, statistics_file_path, "./data/" + statistics_file_path, credentials
)

# Cut file selection
cut_file_numbers = [str(i).zfill(3) for i in range(1, 316)]
with col_cut_file:
    selected_cut_number = st.selectbox(
        "Select Cut File", cut_file_numbers, key="file_selection"
    )

# Check if the selected cut file is on disk, download if necessary
cut_file_path = f"dashboard/{selected_chart.lower()}/{selected_chart.lower()}/c_{selected_chart[-1]}_{selected_cut_number}.csv"


check_and_download(bucket_name, cut_file_path, "./data/" + cut_file_path, credentials)

# Load data and set columns if available
if os.path.exists("./data/" + cut_file_path):
    signal_data = load_csv_data("./data/" + cut_file_path)

    if signal_data is not None:
        signal_data.columns = [
            "Force_X",
            "Force_Y",
            "Force_Z",
            "Vibration_X",
            "Vibration_Y",
            "Vibration_Z",
            "AE_RMS",
            "time",
        ]
        with col_signal:
            signal_to_plot = st.selectbox(
                "Choose Signal/Feature to Plot",
                signal_data.columns[:-1],
                key="signal_feature_selection",
            )


# 4-column layout
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

# 1. Signal Visualization
with col1:
    st.subheader("Signal Visualization")
    # Create a Plotly figure
    fig = go.Figure()

    # Add a line trace for the selected signal
    fig.add_trace(
        go.Scatter(
            x=signal_data["time"],
            y=signal_data[signal_to_plot],
            mode="lines",
            name=signal_to_plot,
        )
    )

    # Customize layout with axis titles and figure title
    fig.update_layout(
        xaxis_title="Time (us)",
        yaxis_title=signal_to_plot,  # Use the selected signal as the y-axis label
        template="plotly_white",
        margin=dict(t=1, l=0, r=0, b=0),
    )

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig, use_container_width=True)

# 2. Frequency Analysis
with col2:
    st.subheader("Frequency Analysis")

    # Perform FFT on the selected signal
    y = np.asarray(signal_data[signal_to_plot])
    n = len(y)
    yf = fft.fft(y)
    xf = fft.fftfreq(n, d=1 / 50000)

    # Create a DataFrame with the frequency and magnitude data
    freq_data = pd.DataFrame(
        {"Frequency (Hz)": xf[: n // 2], "Magnitude": 2.0 / n * np.abs(yf[: n // 2])}
    )

    # Create a Plotly figure for the frequency analysis
    fig = go.Figure()

    # Add a line trace for the frequency data
    fig.add_trace(
        go.Scatter(
            x=freq_data["Frequency (Hz)"],
            y=freq_data["Magnitude"],
            mode="lines",
            name="Magnitude",
        )
    )

    # Customize layout with axis titles
    fig.update_layout(
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude",
        template="plotly_white",
    )

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig, use_container_width=True)

# 3. Target Variable (Wear Flute) vs Cut
with col3:
    st.subheader("Target Variable (Wear Flute) vs Cut")
    wear_data = load_wear_data(selected_chart)
    if wear_data is not None:
        fig = go.Figure()
        for flute in ["flute_1", "flute_2", "flute_3"]:
            fig.add_trace(
                go.Scatter(
                    x=wear_data["cut"],
                    y=wear_data[flute],
                    mode="lines+markers",
                    name=flute.capitalize(),
                )
            )
        fig.update_layout(
            xaxis_title="Cut",
            yaxis_title="Wear [μm]",
            template="plotly_white",
            margin=dict(t=1, l=0, r=0, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

# 4. Correlation Matrix
with col4:
    st.subheader("Correlation Matrix")
    stats_data = load_stats_data(selected_chart)
    main_data = load_main_data(selected_chart)
    signal_to_plot_lower = signal_to_plot.lower()  # Ensure compatibility with lowercase
    if main_data is not None and signal_to_plot_lower in main_data.columns:
        selected_columns = [
            col for col in stats_data.columns if signal_to_plot_lower in col
        ]
        combined_data = pd.concat(
            [
                stats_data[selected_columns],
                wear_data[["flute_1", "flute_2", "flute_3"]],
            ],
            axis=1,
        )
        correlation_matrix = combined_data.corr()

        fig = go.Figure(
            data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns.drop(selected_columns),
                y=correlation_matrix.columns.drop(["flute_1", "flute_2", "flute_3"]),
                colorscale="RdYlBu",
                colorbar=dict(title="Correlation"),
                zmin=-1,
                zmax=1,
                xgap=1,
                ygap=1,
            )
        )
        fig.update_layout(
            xaxis_title="Target Variables",
            yaxis_title="Features",
            margin=dict(t=0, l=0, r=0, b=0),
        )
        st.plotly_chart(fig)
    else:
        st.write(f"Feature {signal_to_plot_lower} not found in main data.")
