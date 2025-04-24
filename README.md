# ***Near Real-Time Total Electron Content (TEC) Over Kenya***

This Streamlit app visualizes hourly Total Electron Content (TEC) over Kenya and automatically updates every 5 minutes. TEC is an important parameter affecting the accuracy of Global Navigation Satellite Systems (GNSS), including GPS. This app downloads Near-Real TEC data, processes it, trains a machine learning model to extraporate TEC values, and displays the extraporated vaues on a map.

## ***Features***

- **Automatic Data Download:** Downloads TEC data from the specified URLs.
- **Data Processing:** Removes outliers and prepares data for training.
- **Model Training:** Trains an ANN model to predict TEC values.
- **Visualization:** Plots the TEC values on a map, showing the predicted TEC over Kenya.
- **Automatic Refresh:** Updates the data and visualization every 5 minutes.

## ***Installation***

### **Running the App Locally**

To run the app locally, follow these steps:

1. **Clone the repository:**
    ```sh
    git clone https://github.com/ngoosojk/tec_app_kenya.git
    cd tec_app_kenya
    ```

2. **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Run the Streamlit app:**
    ```sh
    streamlit run app.py
    ```

## ***Usage***

1. **Run the App:**
    - Start the app using the command `streamlit run app.py`. The app will open in your default web browser.

2. **Automatic Updates:**
    - The app automatically downloads new data every 5 minutes, processes it, trains the model, and updates the visualization.

3. **TEC Visualization:**
    - The map displays the predicted TEC values over Kenya. The color gradient represents the TEC values in TECU (Total Electron Content Units) and the associated Ionospheric range error in meters.


## ***Directory Structure***

- datasets/: Directory for storing downloaded data in JSON format.

- maps/: Directory for saving generated maps.

- app.py: Main application script.

- requirements.txt: File listing all dependencies required to run the app.

- model_performance/: Directory for saving model performance metrics.

- model_losses/: Directory for saving model loss graphs.

- model.keras: Saved model file in Keras format.



## ***Configuration***

- **Data Source URLs:**
    - The app downloads data from the Malindi and Nairobi GNSS receivers, accessible on the ESWUA website (http://www.eswua.ingv.it/index.php/gnss-scintillation/gnss-real-time-data), using specified URLs.

## ***Accessing the Deployed App***

You can access the deployed app at: https://tec-app-kenya.streamlit.app/.
If it’s asleep 😴, just wake it up! 😄


## **Contributing**

Contributions are welcome! If you have any suggestions or improvements, please create a pull request or open an issue.

## **Acknowledgements**

- This app uses data from the Nairobi and Malindi GNSS receivers, owned by the Kenya Space Agency and the Italian Space Agency respectively, accessible on the ESWUA website.
- Special thanks to the Streamlit community for their support and resources.
