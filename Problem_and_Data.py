import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="CNC Predictive Maintenance Dashboard",
    page_icon=":bar_chart:",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    # Predictive maintenance of a CNC milling machine

    In machine cutting processes, in order to ensure surface finish quality, it is imperative to keep the tool 
    used in top operative condition. One approach to monitor the machine wear is to measure certain process parameters, 
    such as cutting force, tool vibration and acoustics emissions.

    ## The data source
    The data was collected from a high speed CNC (computer numerical control) machine (Röders Tech RFM760) cutting stainless 
    steel (HRC52). A platform dynamometer was used for measuring cutting force, three accelerometer was mounted to detect tool 
    vibration in different directions, and a specialised sensor monitored the acoustic emission levels. The outputs of these sensors 
    were captured by corresponding signal conditioning accessories.

    ## Experimental set-up
    """
)

image = Image.open("./assets/exp_setup.png")
st.image(image, caption="1D Convolutional Neural Network", width=600)

st.markdown(
    """
    Experimental set-up showing sensor locations on the high speed CNC milling machine  
    (Source: **[H. Zeng, T. B. Thoe, X. Li and J. Zhou, "Multi-modal Sensing for Machine Health Monitoring in High Speed Machining"](https://ieeexplore.ieee.org/document/4053566)**)

    ## Dataset description
    The raw downloaded data is placed in the `data/raw/` directory and unziped. The data has an internal directory structure:

    ```
    ├── c1 
    │   ├── c1
    │   │   ├── c_1_001.csv
    │   │   ├── c_1_002.csv
    │   │   ├── ...
    │   ├── c1_wear.csv
    ├── c2 
    │   ├── c2
    │   │   ├── c_2_001.csv
    │   │   ├── c_2_002.csv
    │   │   ├── ...
    ├── c3 
    │   ├── c3
    │   │   ├── c_3_001.csv
    │   │   ├── c_3_002.csv
    │   │   ├── ...
    ├── c4 
    │   ├── c4
    │   │   ├── c_4_001.csv
    │   │   ├── c_4_002.csv
    │   │   ├── ...
    │   ├── c4_wear.csv
    ├── c5 
    │   ├── c5
    │   │   ├── c_5_001.csv
    │   │   ├── c_5_002.csv
    │   │   ├── ...
    ├── c6 
    │   ├── c6
    │   │   ├── c_6_001.csv
    │   │   ├── c_6_002.csv
    │   │   ├── ...
    │   ├── c6_wear.csv

    ```

    The data set consists of .csv (comma separated value) files. There are six individual cutter records (folders c1-c6) out of which:
    - c1, c4 and c6 are for training
    - c2, c3 and c5 are for testing

    Each `c$number/c$number` directory corresponds to records for 315 cuts measured by the monitoring system mounted on the CNC milling machine 
    as it removed material off a metal piece. The df the monitoring system recorded was:

    * Column 1: Force (N) in X dimension
    * Column 2: Force (N) in Y dimension
    * Column 3: Force (N) in Z dimension
    * Column 4: Vibration (g) in X dimension
    * Column 5: Vibration (g) in Y dimension
    * Column 6: Vibration (g) in Z dimension
    * Column 7: AE-RMS (V)

    For the training folders (c1,c4,c6), the wear in the flutes of the cutter in units of $10^{-3}$ mm ($\mu m$) is available in the `c$number_wear.csv` files.
    """,
    unsafe_allow_html=True,
)
