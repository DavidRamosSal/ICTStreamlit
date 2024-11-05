import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
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


# st.title("Deep Learning")
st.set_page_config(
    page_title="Deep Learning Modeling",
    layout="wide",
)

# Load image
image = Image.open("./assets/CNN_methods.drawio.png")

left_co, cent_co, last_co = st.columns(3)

with left_co:
    st.markdown("#")

    st.write(
        "Instead of defining features manually, convolutional neural networks can do the work for us."
        + " They process the raw data,creating and optimizing features for prediction."
    )
    st.write(
        "The model we chose consists of: two 1D convolutional layers,"
        + "a maxpool layer, a 1D convolutional layer, a maxpool layer, a droput layer"
        + " and a fully connected layer. The kernel size was set to 3, the amount of filters to 128 and RELU"
        " activation functions were used after each convolutional layer."
    )
    st.write(
        "Training was done using dataset C1, in batches of 5 cuts and for 500 epochs, at a learning rate of 0.002."
    )

with cent_co:
    # Display image in Streamlit app
    st.image(image, caption="1D Convolutional Neural Network", width=600)

DL_file_path = f"dashboard/c1/DL_c1.csv"

check_and_download(bucket_name, DL_file_path, "./data/" + DL_file_path, credentials)

plot_flute_wear_performance(
    "data/dashboard/c1/DL_c1.csv",
    "c1",
    "Performance of the model on training dataset (c1)",
)
plot_flute_wear_performance(
    "data/dashboard/c1/DL_c1.csv",
    "c4",
    "Performance of the model on test dataset (c4)",
)
plot_flute_wear_performance(
    "data/dashboard/c1/DL_c1.csv",
    "c6",
    "Performance of the model on test dataset (c6)",
)
