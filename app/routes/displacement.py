# app/routes/displacement.py
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import pickle
import streamlit as st

class Displacement:
    def __init__(self):
        with open('models/v1/smoothed_displacement_rf_model.pkl', 'rb') as file:
            loaded_model = pickle.load(file)

        self.loaded_model = loaded_model

    def _predict_displacement(self, start_cycle, end_cycle, step_size, velocity):
        cycle_numbers = list(range(start_cycle, end_cycle + 1, step_size))
        new_data = pd.DataFrame({
            'velocity': [velocity] * len(cycle_numbers),
            'cycle_number': cycle_numbers
        })
        predicted_displacement = self.loaded_model.predict(new_data)
        return cycle_numbers, predicted_displacement.tolist()


    def elastic_displacement(self):
        st.title("Elastic Displacement")
        st.write("Use the form below to predict displacement and view the results.")

        # Input fields
        start_cycle = st.number_input("Start Cycle:", min_value=0, value=0, step=50)
        end_cycle = st.number_input("End Cycle:", min_value=1000, value=50000, step=50)
        step_size = st.number_input("Step Size:", min_value=100, value=100, step=50)
        velocity = st.selectbox("Velocity (km/h):", options=[160, 210, 270, 320])

        if st.button("Predict"):
            if start_cycle < end_cycle:
                # Perform predictions
                cycle_numbers, predicted_displacement = self._predict_displacement(start_cycle, end_cycle, step_size, velocity)

                # Display results in a table
                st.subheader("Predicted Displacement Table")
                result_df = pd.DataFrame({
                    "Cycle Number": cycle_numbers,
                    "Predicted Displacement": [f"{value:.9f}" for value in predicted_displacement]
                })
                st.dataframe(result_df)

                # Display results as a graph
                st.subheader("Predicted Displacement Graph")
                graph_image = self.plot_displacement_high_res(start_cycle, end_cycle, step_size, velocity)
                st.image(graph_image, caption="Predicted Displacement", use_container_width=True)
            else:
                st.error("Start Cycle must be less than End Cycle.")

    def plot_displacement_high_res(self, start_cycle, end_cycle, step_size, velocity):
        cycle_numbers, predicted_displacement = self._predict_displacement(start_cycle, end_cycle, step_size, velocity)

        plt.figure(figsize=(12, 8), dpi=300)  # Increase resolution with dpi and size
        plt.plot(cycle_numbers, predicted_displacement, label='Predicted Displacement', color='blue')
        plt.xlabel('Cycle Number')
        plt.ylabel('Displacement')
        plt.title(f'Displacement Prediction for Velocity {velocity} km/h')
        plt.grid(True)
        plt.legend()

        buf = BytesIO()
        plt.savefig(buf, format="PNG", dpi=300)  # Save with high dpi
        plt.close()
        buf.seek(0)
        return Image.open(buf)


    def _generate_and_plot_displacement_multi_aligned(self, velocity_ranges, step_size):
        """
        Generates and plots multi-aligned displacement predictions.

        Parameters:
            model: Trained machine learning model for displacement.
            velocity_ranges: List of velocity ranges as [[velocity, start_cycle, end_cycle], ...].
            step_size: Step size for cycle generation.

        Returns:
            BytesIO: Buffer containing the plot image.
        """
        previous_end_value = None

        model = self.loaded_model

        # Define a color map for velocities
        color_map = {velocity: color for velocity, color in zip(
            sorted(set([v[0] for v in velocity_ranges])),
            ['orange', 'green', 'red', 'purple', 'cyan', 'magenta']
        )}

        plt.figure(figsize=(12, 8))

        for velocity, start_cycle, end_cycle in velocity_ranges:
            cycle_numbers = list(range(start_cycle, end_cycle + 1, step_size))
            new_data = pd.DataFrame({
                'velocity': [velocity] * len(cycle_numbers),
                'cycle_number': cycle_numbers
            })

            # Predict displacement
            predicted_displacement = model.predict(new_data)

            # Adjust predictions for continuity
            if previous_end_value is not None:
                adjustment = previous_end_value - predicted_displacement[0]
                predicted_displacement += adjustment

            # Update the end value for the next velocity
            previous_end_value = predicted_displacement[-1]

            # Plot displacement predictions
            plt.plot(cycle_numbers, predicted_displacement,
                    label=f'Predicted Displacement (Velocity {velocity} km/h)',
                    color=color_map[velocity], linestyle='-')

        # Plot settings
        plt.xlabel('Cycle Number')
        plt.ylabel('Displacement (mm)')
        plt.title('Aligned Displacement Predictions')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Save plot to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return buf



    def displacement_multi_aligned(self, cell_style):
        st.subheader("Displacement Multi-Aligned Prediction")

        st.markdown(cell_style, unsafe_allow_html=True)

        # User inputs for number of ranges and step size
        num_ranges = st.number_input("Enter number of velocity ranges:", min_value=1, value=3, step=1)
        step_size = st.number_input("Enter step size:", min_value=1, value=1000, step=1)

        velocity_options = [160, 210, 270, 320, 380]  # Predefined velocity options

        velocity_ranges = []
        st.write("Enter the velocity ranges:")

        # Create rows and columns dynamically based on the number of ranges
        rows = (num_ranges + 1) // 2  # Determine number of rows for a 2-column layout
        previous_end_cycle = 0  # Initialize previous_end_cycle

        for row in range(rows):
            cols = st.columns(2)  # Create 2 columns per row
            for col_idx, col in enumerate(cols):
                range_idx = row * 2 + col_idx
                if range_idx < num_ranges:
                    with col:
                        st.markdown(
                            f"""
                            <div style="border-bottom: 1px solid #d3d3d3; margin-bottom: 10px; padding: 10px;">
                                <p style="font-weight: bold;">Range {range_idx + 1}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        velocity = st.selectbox(f"Velocity (km/h) for range {range_idx + 1}", options=velocity_options, key=f"velocity_{range_idx}")

                        if range_idx == 0:
                            start_cycle = st.number_input(
                                f"Start cycle for range {range_idx + 1}",
                                min_value=0,
                                value=0,
                                step=1000,
                                key=f"start_{range_idx}"
                            )
                        else:
                            st.number_input(
                                f"Start cycle for range {range_idx + 1} (Automatically Set to {previous_end_cycle})",
                                value=previous_end_cycle,
                                disabled=True,
                                key=f"start_{range_idx}"
                            )
                            start_cycle = previous_end_cycle

                        end_cycle = st.number_input(
                            f"End cycle for range {range_idx + 1}",
                            min_value=start_cycle + 1,
                            value=start_cycle + 10000,
                            step=1000,
                            key=f"end_{range_idx}"
                        )

                        velocity_ranges.append([velocity, start_cycle, end_cycle])
                        previous_end_cycle = end_cycle

        if st.button("Generate Predictions", key="generate_predictions"):
            plot_buffer = self._generate_and_plot_displacement_multi_aligned(velocity_ranges=velocity_ranges, step_size=step_size)
            st.image(plot_buffer, caption='Displacement Predictions', use_container_width=True)

            # Generate and display table
            combined_results = []
            for velocity, start_cycle, end_cycle in velocity_ranges:
                cycle_numbers = list(range(start_cycle, end_cycle + 1, step_size))
                new_data = pd.DataFrame({'velocity': [velocity] * len(cycle_numbers), 'cycle_number': cycle_numbers})
                predicted_displacement = self.loaded_model.predict(new_data)
                combined_results.extend(zip(new_data['velocity'], new_data['cycle_number'], predicted_displacement))

            result_df = pd.DataFrame(combined_results, columns=["Velocity (km/h)", "Cycle Number", "Smoothed Displacement"])
            st.dataframe(result_df)
