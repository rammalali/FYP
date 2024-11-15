# app/routes/displacement.py
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import pickle

# Load the pre-trained model
with open('data/models/RandomForestRegressor.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

def predict_displacement(start_cycle, end_cycle, step_size, velocity):
    cycle_numbers = list(range(start_cycle, end_cycle + 1, step_size))
    new_data = pd.DataFrame({
        'velocity': [velocity] * len(cycle_numbers),
        'Cycle_Number': cycle_numbers
    })
    predicted_displacement = loaded_model.predict(new_data)
    return cycle_numbers, predicted_displacement.tolist()

def plot_displacement(start_cycle, end_cycle, step_size, velocity):
    cycle_numbers, predicted_displacement = predict_displacement(start_cycle, end_cycle, step_size, velocity)
    
    plt.figure(figsize=(6, 4))
    plt.plot(cycle_numbers, predicted_displacement, label='Predicted Displacement', color='blue')
    plt.xlabel('Cycle Number')
    plt.ylabel('Displacement')
    plt.title(f'Displacement Prediction for Velocity {velocity} km/h')
    plt.grid(True)
    plt.legend()
    
    buf = BytesIO()
    plt.savefig(buf, format="PNG", dpi=80)
    plt.close()
    buf.seek(0)
    return Image.open(buf)
