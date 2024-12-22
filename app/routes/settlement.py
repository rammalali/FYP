# app/routes/settlement.py
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import pickle
import streamlit as st


with open('models/v1/ps_rf_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

def _predict_settlement(start_cycle, end_cycle, step_size, velocity):
    cycle_numbers = list(range(start_cycle, end_cycle + 1, step_size))
    new_data = pd.DataFrame({
        'velocity': [velocity] * len(cycle_numbers),
        'Cycle_Number': cycle_numbers
    })
    predicted_settlement = loaded_model.predict(new_data)
    return cycle_numbers, predicted_settlement.tolist()


def permanent_settlement():
    st.title("Permanent Settlement")
    st.write("Use the form below to predict settlement and view the results.")

    # Input fields
    start_cycle = st.number_input("Start Cycle:", min_value=0, value=0, step=1)
    end_cycle = st.number_input("End Cycle:", min_value=1, value=10, step=1)
    step_size = st.number_input("Step Size:", min_value=1, value=1, step=1)
    velocity = st.selectbox("Velocity (km/h):", options=[160, 210, 270, 320])

    if st.button("Predict"):
        if start_cycle < end_cycle:
            # Perform predictions
            cycle_numbers, predicted_settlement = _predict_settlement(start_cycle, end_cycle, step_size, velocity)

            # Display results in a table
            st.subheader("Predicted Settlement Table")
            result_df = pd.DataFrame({
                "Cycle Number": cycle_numbers,
                "Predicted Settlement": [f"{value:.9f}" for value in predicted_settlement]
            })
            st.dataframe(result_df)

            # Display results as a graph
            st.subheader("Predicted Settlement Graph")
            graph_image = plot_settlement_high_res(start_cycle, end_cycle, step_size, velocity)
            st.image(graph_image, caption="Predicted Settlement", use_container_width=True)
        else:
            st.error("Start Cycle must be less than End Cycle.")

def plot_settlement_high_res(start_cycle, end_cycle, step_size, velocity):
    cycle_numbers, predicted_settlement = _predict_settlement(start_cycle, end_cycle, step_size, velocity)

    plt.figure(figsize=(12, 8), dpi=300)  # Increase resolution with dpi and size
    plt.plot(cycle_numbers, predicted_settlement, label='Predicted Settlement', color='blue')
    plt.xlabel('Cycle Number')
    plt.ylabel('Settlement')
    plt.title(f'Settlement Prediction for Velocity {velocity} km/h')
    plt.grid(True)
    plt.legend()

    buf = BytesIO()
    plt.savefig(buf, format="PNG", dpi=300)  # Save with high dpi
    plt.close()
    buf.seek(0)
    return Image.open(buf)
