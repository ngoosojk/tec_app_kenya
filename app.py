import os
import json
import urllib.request
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import psutil
import pytz

# Create directories if they don't exist
os.makedirs('datasets', exist_ok=True)
os.makedirs('maps', exist_ok=True)

# Function to download data from a given URL and return the filename
def download_data(url, filename):
    try:
        webURL = urllib.request.urlopen(url)
        param = json.loads(webURL.read().decode())
        parameters = param["records"]

        # Create a JSON file with the data in the datasets folder
        filepath = os.path.join('datasets', filename)
        with open(filepath, 'w') as sample:
            json.dump(parameters, sample, indent=4, separators=(',', ': '))

        print(f"Data downloaded and saved to {filepath}")
        return filepath

    except Exception as e:
        print(f"Failed to download data from {url}. The servers cannot be reached. Error: {e}")
        return None

# Function to remove outliers using the IQR method and a minimum threshold for vtec values
def remove_outliers(ipp_lons, ipp_lats, vtec_values, min_vtec_threshold=1.0):
    valid_indices = [i for i, v in enumerate(vtec_values) if v is not None]
    filtered_ipp_lons = [ipp_lons[i] for i in valid_indices]
    filtered_ipp_lats = [ipp_lats[i] for i in valid_indices]
    filtered_vtec_values = [vtec_values[i] for i in valid_indices]

    if len(filtered_vtec_values) == 0:
        print("No valid VTEC values found after filtering.")
        return np.array([]), np.array([]), np.array([])

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
def train_model(X_train, y_train, model_path='model.keras'):
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

def plot_data(filenames, model_path='model.keras'):
    ipp_lons = []
    ipp_lats = []
    vtec_values = []

    for filename in filenames:
        if filename is None:
            print("Skipping plot due to failed data download.")
            return

        with open(filename, 'r') as f:
            data = json.load(f)
        ipp_lons.extend([record['ipp_lon'] for record in data if record['elevation'] is not None and record['elevation'] >= 30])
        ipp_lats.extend([record['ipp_lat'] for record in data if record['elevation'] is not None and record['elevation'] >= 30])
        vtec_values.extend([record['vtec'] for record in data if record['elevation'] is not None and record['elevation'] >= 30])

    filtered_ipp_lons, filtered_ipp_lats, filtered_vtec_values = remove_outliers(ipp_lons, ipp_lats, vtec_values)

    if len(filtered_vtec_values) == 0:
        print("No valid data to plot after removing outliers.")
        return

    X_train, y_train = prepare_data(filtered_ipp_lons, filtered_ipp_lats, filtered_vtec_values)

    # Train a new model each time
    model = train_model(X_train, y_train, model_path)

    min_latitude = -5.0
    max_latitude = 5.2
    min_longitude = 33.0
    max_longitude = 42.0

    grid_lons, grid_lats = np.meshgrid(np.linspace(min_longitude, max_longitude, 100), np.linspace(min_latitude, max_latitude, 100))
    grid_vtec = predict_vtec(model, grid_lons.flatten(), grid_lats.flatten())

    fig, ax = plt.subplots(figsize=(8, 7))  # Increase the size of the map
    m = Basemap(projection='merc', llcrnrlat=min_latitude, urcrnrlat=max_latitude,
                llcrnrlon=min_longitude, urcrnrlon=max_longitude,
                resolution='l', ax=ax)
    m.drawcoastlines()
    m.drawcountries()

    # Compute label positions with adjusted offsets
    label_offset_lat = 0.3  # Increased offset for latitude labels
    label_offset_lon = 0.2  # Reduced offset for longitude labels
    parallels = np.arange(min_latitude, max_latitude, 5.0)
    meridians = np.arange(min_longitude, max_longitude, 4.0)

    # Add latitude labels with °N, °S, or 0°
    for lat in parallels:
        x, y = m(min_longitude - label_offset_lat, lat)  # Left label offset
        if lat == 0:
            label = '0°'
        else:
            direction = '°N' if lat > 0 else '°S'
            label = f'{abs(lat)}{direction}'
        plt.text(x, y, label, ha='right', va='center', fontsize=11, color='black')

    # Add longitude labels with °E, °W, or 0°
    for lon in meridians:
        x, y = m(lon, min_latitude - label_offset_lon)  # Bottom label offset
        if lon == 0:
            label = '0°'
        else:
            direction = '°E' if lon > 0 else '°W'
            label = f'{abs(lon)}{direction}'
        plt.text(x, y, label, ha='center', va='top', fontsize=11, color='black')

    x, y = m(grid_lons, grid_lats)
    x = x.reshape(grid_lons.shape)
    y = y.reshape(grid_lats.shape)
    sc = m.scatter(x, y, c=grid_vtec.flatten(), cmap='viridis', marker='o', vmin=0, vmax=100)

    # Add dotted contour lines at whole number intervals
    contour_levels = np.arange(np.floor(grid_vtec.min()), np.ceil(grid_vtec.max()) + 1, 5)
    cs = m.contour(x, y, grid_vtec.reshape(grid_lons.shape), levels=contour_levels, colors='white', linewidths=2.0, linestyles='dotted')
    plt.clabel(cs, inline=1, fontsize=8, fmt='%1.0f')

    # Add the main colorbar for TECU
    cbar = plt.colorbar(sc, ax=ax, fraction=0.036, pad=0.1)
    cbar.set_label('TECU', labelpad=-50, y=0.5, rotation=90, ha='center')

    # Calculate ionospheric range error (L1) values
    ionospheric_error = (3.24 / 20) * grid_vtec.flatten()

    # Create a second colorbar axis for Ionospheric range error (L1)/m
    cbar2_ax = cbar.ax.twinx()  # Create a twin of the main colorbar axis
    cbar2_ax.set_ylim(sc.get_clim())  # Set the limits of cbar2 to match the main colorbar
    cbar2_ax.set_yticks(cbar.get_ticks())  # Set the ticks of cbar2 to match the main colorbar
    cbar2_ax.set_yticklabels([(3.24 / 20) * t for t in cbar.get_ticks()])  # Set the tick labels for cbar2
    cbar2_ax.set_ylabel('Ionospheric range error (L1) in m', fontsize=12)  # Set the label for cbar2

    start_time_str = filenames[0].split('_')[1] + ' ' + filenames[0].split('_')[2] + ':' + filenames[0].split('_')[3] + ':' + filenames[0].split('_')[4]
    end_time_str = filenames[0].split('_')[5] + ' ' + filenames[0].split('_')[6] + ':' + filenames[0].split('_')[7] + ':' + filenames[0].split('_')[8].split('.')[0]
    plt.suptitle(f'Near Real-Time Hourly Total Electron Content (TEC) Over Kenya\n{start_time_str} to {end_time_str} UTC', fontsize=15)

    # Save the map as an image file in the maps folder
    map_filename = os.path.join('maps', f"map_{start_time_str.replace(' ', '_').replace(':', '-')}_to_{end_time_str.replace(' ', '_').replace(':', '-')}.png")
    plt.savefig(map_filename, bbox_inches='tight')  # Use bbox_inches='tight' to ensure proper alignment of ticks

    plt.tight_layout()  # Adjust subplots to fit into figure area, leaving space for the title.
    st.pyplot(fig)

# Function to automate the process every 5 minutes
@st.cache_data(ttl=240)
def schedule_download_and_plot():
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=1)
    st_time = start_time.strftime("%Y-%m-%d%%20%H:%M:%S")
    en_time = end_time.strftime("%Y-%m-%d_%H:%M:%S")
    
    nai_url = f'http://ws-eswua.rm.ingv.it/scintillation.php/records/wsnai0p?filter=dt,bt,{st_time},{en_time}&filter0=PRN,sw,&filter1=PRN,sw,N&filter2=PRN,sw,N&filter3=PRN,sw,N&filter4=PRN,sw,N&filter5=PRN,sw,N&filter6=PRN,sw,N&include=dt,PRN,vtec,ipp_lon,ipp_lat,elevation,&order=dt'
    mal_url = f'http://ws-eswua.rm.ingv.it/scintillation.php/records/wsmal0p?filter=dt,bt,{st_time},{en_time}&filter0=PRN,sw,&filter1=PRN,sw,N&filter2=PRN,sw,N&filter3=PRN,sw,N&filter4=PRN,sw,N&filter5=PRN,sw,N&filter6=PRN,sw,N&include=dt,PRN,vtec,ipp_lon,ipp_lat,elevation,&order=dt'

    nai_filename = download_data(nai_url, f'nai0p_{start_time.strftime("%Y-%m-%d_%H_%M_%S")}_{end_time.strftime("%Y-%m-%d_%H_%M_%S")}.json')
    mal_filename = download_data(mal_url, f'mal0p_{start_time.strftime("%Y-%m-%d_%H_%M_%S")}_{end_time.strftime("%Y-%m-%d_%H_%M_%S")}.json')

    plot_data([nai_filename, mal_filename], model_path='model.keras')

# Streamlit app layout and execution
st.title("Near Real-Time Total Electron Content (TEC) Over Kenya")
st.write("This app visualizes hourly TEC over Kenya and automatically updates every 5 minutes.")
st_autorefresh(interval=300000, key="data_refresh")
schedule_download_and_plot()

# Set the interval (1 hour)
REBOOT_INTERVAL = 3600  # in seconds

# Initialize the next reboot time in session state
if "next_reboot_time" not in st.session_state:
    st.session_state.next_reboot_time = datetime.now() + timedelta(seconds=REBOOT_INTERVAL)

# Display a message
st.write(f"This app visualizes hourly TEC over Kenya and automatically refreshes every 5 minutes. The app is set to reboot every {REBOOT_INTERVAL // 60} minutes ( 1 Hr) to ensure efficient performance.")

# Check if it's time to reboot
if datetime.now() >= st.session_state.next_reboot_time:
    st.write("Rebooting the app now...")
    st.session_state.next_reboot_time = datetime.now() + timedelta(seconds=REBOOT_INTERVAL)  # Reset the reboot time
    st.rerun()  # Trigger the app reboot

# Normal app logic
# Convert to UTC
next_reboot_time_utc = st.session_state.next_reboot_time.astimezone(pytz.UTC)
# Display time in UTC
st.write(f"Next reboot scheduled at: {next_reboot_time_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")

# Function to check memory usage and return the memory usage percentage
def print_memory_usage():
    # Get memory usage in MB
    memory_info = psutil.virtual_memory()
    memory_usage = memory_info.percent
    total_memory = memory_info.total / (1024 ** 3)  # in GB
    used_memory = memory_info.used / (1024 ** 3)  # in GB
    
    print(f"Total Memory: {total_memory:.2f} GB")
    print(f"Used Memory: {used_memory:.2f} GB")
    print(f"Memory Usage: {memory_usage:.2f}%")
    
    return memory_usage  # Return the memory usage for further use

# Call the function and get the memory usage
memory_usage = print_memory_usage()

# Example threshold for high memory usage
if memory_usage > 70:  # Example threshold
    st.warning("High memory usage detected. Consider optimizing the app.")
