import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data_extraction import bucket_name, credentials
from data_extraction import check_and_download


def plot_flute_wear_performance(data_path, cols_prefix, title):
    dl_df = pd.read_csv(data_path)

    metrics = []
    for i in range(1, 4):
        mse = np.mean((dl_df[f"{cols_prefix}w{i}"] - dl_df[f"{cols_prefix}w{i}p"]) ** 2)
        mape = (
            np.mean(
                np.abs(
                    (dl_df[f"{cols_prefix}w{i}"] - dl_df[f"{cols_prefix}w{i}p"])
                    / dl_df[f"{cols_prefix}w{i}"]
                )
            )
            * 100
        )
        metrics.append((mse, mape))

    # Prepare plot data
    x = list(range(1, 316))
    y_data = [
        (dl_df[f"{cols_prefix}w{i}"], dl_df[f"{cols_prefix}w{i}p"]) for i in range(1, 4)
    ]

    fig = make_subplots(
        rows=1, cols=3, subplot_titles=("Flute 1", "Flute 2", "Flute 3")
    )

    # annotations for each flute
    for i, ((y_real, y_pred), (mse, mape)) in enumerate(zip(y_data, metrics), start=1):
        fig.add_trace(
            go.Scatter(
                x=x, y=y_real, mode="lines+markers", name=f"Real flute {i} wear"
            ),
            row=1,
            col=i,
        )
        fig.add_trace(
            go.Scatter(
                x=x, y=y_pred, mode="lines+markers", name=f"Predicted flute {i} wear"
            ),
            row=1,
            col=i,
        )

        # Add MSE and MAPE as annotations
        fig.add_annotation(
            x=0.18 + 0.41 * (i - 1),
            y=0.2,
            xref="paper",
            yref="paper",
            text=f"MSE: {round(mse, 4)}<br>MAPE: {round(mape, 2)}%",
            showarrow=False,
            font=dict(size=15),
            bgcolor=None,
            bordercolor="black",
        )

    fig.update_layout(
        height=500,
        width=None,
        title_text=title,
        showlegend=True,
        # annotations=annotations,
    )

    for i in range(1, 4):
        fig.update_xaxes(title_text="Cut number", row=1, col=i)
        fig.update_yaxes(title_text="wear (um)", row=1, col=i)

    st.plotly_chart(fig, use_container_width=True, key=cols_prefix)


st.set_page_config(
    page_title="Machine Learning Modeling",
    layout="wide",
)
st.write("### About the Model:")

col1, col2 = st.columns(2)
with col1:
    st.write("#### Features Used:")
    st.markdown(
        """ 
        - Force statistics (X, Y and Z)
        - Vibration statistics (X, Y and Z)
        - AE_RMS statistics
    """
    )
    st.write("Statistics calculated: minimum, maximum, mean value, ")
    st.write("std, zero crossings, kurtosis, skewness and energy.")

with col2:
    st.write("#### Lasso Linear Regression Model:")
    st.markdown(
        """
        - A linear model that estimates the coefficients of linear relationships.
        - It uses L1 regularization to prevent overfitting.
        - Suitable for high-dimensional datasets.
        - Can perform variable selection.
    """
    )
    st.write("Wrapped in a Multi-output regression model.")
    st.write("This allows for having multiple targets (3 flutes) at the same time.")

# Dropdown selector for model selection
col1, col2, col3 = st.columns(3)
with col1:
    st.write(" ")
with col2:
    selected_model = st.radio(
        "Select the best model trained with dataset",
        ["C1", "C4", "C6"],
        key="model_selection",
        horizontal=True,
    ).lower()
with col1:
    st.write(" ")

# Load data based on selected model
st.write(f"## {selected_model.upper()} Dataset Analysis")

ML_file_path = f"dashboard/{selected_model.lower()}/ML_{selected_model.lower()}.csv"

check_and_download(bucket_name, ML_file_path, "./data/" + ML_file_path, credentials)

datasets = [selected_model] + [
    dataset for dataset in ["c1", "c4", "c6"] if dataset != selected_model
]

for dataset in datasets:
    if dataset == selected_model:
        dataset_type = "training"
    else:
        dataset_type = "test"
    plot_flute_wear_performance(
        f"data/dashboard/{selected_model}/ML_{selected_model}.csv",
        f"{dataset}",
        f"Performance of the model on {dataset_type} dataset ({dataset})",
    )
