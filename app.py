import json
import urllib.request
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

# Create directories if they don't exist
os.makedirs('datasets', exist_ok=True)
os.makedirs('maps', exist_ok=True)

# Function to download data from a given URL and return the filename
def download_data(url, filename):
    webURL = urllib.request.urlopen(url)
    param = json.loads(webURL.read().decode())
    parameters = param["records"]

    # Create a JSON file with the data in the datasets folder
    filepath = os.path.join('datasets', filename)
    with open(filepath, 'w') as sample:
        json.dump(parameters, sample, indent=4, separators=(',', ': '))

    print(f"Data downloaded and saved to {filepath}")
    return filepath

# Function to remove outliers using the IQR method and a minimum threshold for vtec values
def remove_outliers(ipp_lons, ipp_lats, vtec_values, min_vtec_threshold=1.0):
    valid_indices = [i for i, v in enumerate(vtec_values) if v is not None]
    filtered_ipp_lons = [ipp_lons[i] for i in valid_indices]
    filtered_ipp_lats = [ipp_lats[i] for i in valid_indices]
    filtered_vtec_values = [vtec_values[i] for i in valid_indices]

    Q1 = np.percentile(filtered_vtec_values, 25)
    Q3 = np.percentile(filtered_vtec_values, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 0.5 * IQR
    upper_bound = Q3 + 0.5 * IQR

    outliers = (np.array(filtered_vtec_values) < lower_bound) | (np.array(filtered_vtec_values) > upper_bound) | (np.array(filtered_vtec_values) < min_vtec_threshold)
    final_ipp_lons = np.array(filtered_ipp_lons)[~outliers]
    final_ipp_lats = np.array(filtered_ipp_lats)[~outliers]
    final_vtec_values = np.array(filtered_vtec_values)[~outliers]

    return final_ipp_lons, final_ipp_lats, final_vtec_values

# Function to prepare data for training
def prepare_data(ipp_lons, ipp_lats, vtec_values):
    X = np.column_stack((ipp_lons, ipp_lats))
    y = np.array(vtec_values)
    return X, y
    
# Function to train the model with early stopping
def train_model(X_train, y_train, model_path='model.h5'):
    if os.path.exists(model_path):
        model = load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss='mean_absolute_error')
        print("Loaded existing model.")
    else:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(2,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_absolute_error')
        print("Created new model.")

    # Implement early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=0, callbacks=[early_stopping])
    print("Training loss:", history.history['loss'][-1])
    model.save(model_path)
    print("Model saved.")
    return model

# Function to predict VTEC values using the trained model
def predict_vtec(model, lons, lats):
    X_pred = np.column_stack((lons, lats))
    vtec_pred = model.predict(X_pred)
    return vtec_pred

# Function to plot data from the JSON files and predicted VTEC values
def plot_data(filenames, model):
    ipp_lons = []
    ipp_lats = []
    vtec_values = []

    for filename in filenames:
        with open(filename, 'r') as f:
            data = json.load(f)
        ipp_lons.extend([record['ipp_lon'] for record in data if record['elevation'] is not None and record['elevation'] >= 30])
        ipp_lats.extend([record['ipp_lat'] for record in data if record['elevation'] is not None and record['elevation'] >= 30])
        vtec_values.extend([record['vtec'] for record in data if record['elevation'] is not None and record['elevation'] >= 30])

    filtered_ipp_lons, filtered_ipp_lats, filtered_vtec_values = remove_outliers(ipp_lons, ipp_lats, vtec_values)
    X_train, y_train = prepare_data(filtered_ipp_lons, filtered_ipp_lats, filtered_vtec_values)
    model = train_model(X_train, y_train)

    min_latitude = -5.0
    max_latitude = 5.1
    min_longitude = 33.5
    max_longitude = 42.0

    grid_lons, grid_lats = np.meshgrid(np.linspace(min_longitude, max_longitude, 100), np.linspace(min_latitude, max_latitude, 100))
    grid_lons = grid_lons.flatten()
    grid_lats = grid_lats.flatten()
    grid_vtec = predict_vtec(model, grid_lons, grid_lats)

    fig, ax = plt.subplots(figsize=(10, 5))
    m = Basemap(projection='merc', llcrnrlat=min_latitude, urcrnrlat=max_latitude,
                llcrnrlon=min_longitude, urcrnrlon=max_longitude,
                resolution='l', ax=ax)
    m.drawcoastlines()
    m.drawcountries()
    x, y = m(grid_lons, grid_lats)
    sc = m.scatter(x, y, c=grid_vtec, cmap='viridis', marker='o', vmin=0, vmax=100)

    plt.colorbar(sc, label='VTEC')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    start_time_str = filenames[0].split('_')[1] + ' ' + filenames[0].split('_')[2] + ':' + filenames[0].split('_')[3] + ':' + filenames[0].split('_')[4]
    end_time_str = filenames[0].split('_')[5] + ' ' + filenames[0].split('_')[6] + ':' + filenames[0].split('_')[7] + ':' + filenames[0].split('_')[8].split('.')[0]
    plt.title(f'IPP Latitude vs Longitude with VTEC values\n{start_time_str} to {end_time_str} UTC')

    # Save the map as an image file in the maps folder
    map_filename = os.path.join('maps', f"map_{start_time_str.replace(' ', '_').replace(':', '-')}_to_{end_time_str.replace(' ', '_').replace(':', '-')}.png")
    plt.savefig(map_filename)

    plt.tight_layout()
    st.pyplot(fig)

# Function to automate the process every 5 minutes
@st.cache_data(ttl=240)
def schedule_download_and_plot():
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=1)
    st_time = start_time.strftime("%Y-%m-%d%%20%H:%M:%S")
    en_time = end_time.strftime("%Y-%m-%d_%H:%M:%S")

    nai_url = f'http://ws-eswua.rm.ingv.it/scintillation.php/records/wsnai0p?filter=dt,bt,{st_time},{en_time}&filter0=PRN,sw,&filter1=PRN,sw,N&filter2=PRN,sw,N&filter3=PRN,sw,N&filter4=PRN,sw,N&filter5=PRN,sw,N&filter6=PRN,sw,N&include=dt,PRN,vtec,ipp_lon,ipp_lat,elevation,&order=dt'
    kil_url = f'http://ws-eswua.rm.ingv.it/scintillation.php/records/wsmal0p?filter=dt,bt,{st_time},{en_time}&filter0=PRN,sw,&filter1=PRN,sw,N&filter2=PRN,sw,N&filter3=PRN,sw,N&filter4=PRN,sw,N&filter5=PRN,sw,N&filter6=PRN,sw,N&include=dt,PRN,vtec,ipp_lon,ipp_lat,elevation,&order=dt'

    nai_filename = f'nai0p_{start_time.strftime("%Y-%m-%d_%H_%M_%S")}_{end_time.strftime("%Y-%m-%d_%H_%M_%S")}.json'
    kil_filename = f'kil0p_{start_time.strftime("%Y-%m-%d_%H_%M_%S")}_{end_time.strftime("%Y-%m-%d_%H_%M_%S")}.json'

    nai_filepath = download_data(nai_url, nai_filename)
    kil_filepath = download_data(kil_url, kil_filename)

    # Prepare data and train the model
    ipp_lons = []
    ipp_lats = []
    vtec_values = []

    for filepath in [nai_filepath, kil_filepath]:
        with open(filepath, 'r') as f:
            data = json.load(f)
        ipp_lons.extend([record['ipp_lon'] for record in data if record['elevation'] is not None and record['elevation'] >= 30])
        ipp_lats.extend([record['ipp_lat'] for record in data if record['elevation'] is not None and record['elevation'] >= 30])
        vtec_values.extend([record['vtec'] for record in data if record['elevation'] is not None and record['elevation'] >= 30])

    filtered_ipp_lons, filtered_ipp_lats, filtered_vtec_values = remove_outliers(ipp_lons, ipp_lats, vtec_values)
    X_train, y_train = prepare_data(filtered_ipp_lons, filtered_ipp_lats, filtered_vtec_values)
    model = train_model(X_train, y_train)

    plot_data([nai_filepath, kil_filepath], model)

# Streamlit app layout and execution
st.title("Near Real-Time Total Electron Content (TEC) Over Kenya ")
st.write("This app visualizes hourly TEC Over Kenya, updated after 5 mins.")
st_autorefresh(interval=60000, key="data_refresh")
schedule_download_and_plot()