import pandas as pd

# Initialize data
data = {
    'Name': ['Sravan', 'Deepak', 'Radha', 'Vani'],
    'College': ['vignan', 'vignan Lara', 'vignan', 'vignan'],
    'Department': ['CSE', 'IT', 'IT', 'CSE'],
    'Profession': ['Student', 'Assistant Professor', 'Programmer & ass. Proff', 'Programmer & Scholar'],
    'Age': [22, 32, 45, 37]
}

# Create DataFrame
df = pd.DataFrame(data)

# Add metadata
metadata = {
    'scale': 0.1,
    'offset': 15,
    'instrument_name': 'Binky'
}

# Store DataFrame and metadata in HDF5
path = "src/simple_tests/files/college_data.hdf5"
with pd.HDFStore(path, mode='w') as store:
    store.put('data', df)  # Store the DataFrame
    store.get_storer('data').attrs.metadata = metadata  # Add metadata

# Retrieve DataFrame and metadata
with pd.HDFStore(path, mode='r') as store:
    loaded_df = store['data']  # Load the DataFrame
    loaded_metadata = store.get_storer('data').attrs.metadata  # Load metadata

# Display results
print("Loaded DataFrame:\n", loaded_df)
print("\nLoaded Metadata:\n", loaded_metadata)




##### Make requests 

data = {
    "lat": [63, 63, 64, 64],
    "lon": [10, 11, 10, 11]
 }


df = pd.DataFrame(data)
df_metadat = {
    "dxdy": 160,
    "prediction_mode": "area", # Can be "area" or "point"
    "time": [0, 3600], # The times to predict for 
    "plot_predictions": True,
    "plot_path": "figures/tests/RunModel/"
}

with pd.HDFStore(path, mode="w") as store:
    store.put("prediction_requests", df)
    store.get_storer("prediction_requests").attrs.metadata = df_metadat

# Retrieve DataFrame and metadata
with pd.HDFStore(path, mode='r') as store:
    loaded_df = store['prediction_requests']  # Load the DataFrame
    loaded_metadata = store.get_storer('prediction_requests').attrs.metadata  # Load metadata
# Display results
print("Loaded DataFrame:\n", loaded_df)
print("\nLoaded Metadata:\n", loaded_metadata)