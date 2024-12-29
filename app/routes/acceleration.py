import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
from PIL import Image
import pickle


from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.pipeline import Pipeline
import numpy as np

class Acceleration:
    def __init__(self, n_max_rf_model, p_max_rf_model):
        self.n_max_rf_model = n_max_rf_model
        self.p_max_rf_model = p_max_rf_model

    def set_n_max_rf_model(self, n_max_rf_model):
        self.n_max_rf_model = n_max_rf_model

    def set_p_max_rf_model(self, p_max_rf_model):
        self.p_max_rf_model = p_max_rf_model



    # Function to generate and plot accelerations
    def generate_and_plot_accelerations(self, start_cycle, end_cycle, step_size, velocity):
        cycle_numbers = list(range(start_cycle, end_cycle + 1, step_size))
        new_data = pd.DataFrame({
            'velocity': [velocity] * len(cycle_numbers),
            'Cycle_Number': cycle_numbers
        })
        neg_model = self.n_max_rf_model
        pos_model = self.p_max_rf_model

        # Predict negative and positive max acceleration
        predicted_neg_acceleration = neg_model.predict(new_data)
        predicted_pos_acceleration = pos_model.predict(new_data)

        # Plot
        plt.figure(figsize=(10, 6), dpi=300)
        plt.plot(cycle_numbers, predicted_neg_acceleration, label='Predicted Negative Max Acceleration', color='blue')
        plt.plot(cycle_numbers, predicted_pos_acceleration, label='Predicted Positive Max Acceleration', color='red')

        plt.xlabel('Cycle Number')
        plt.ylabel('Acceleration (m/s²)')
        plt.title(f'Acceleration Prediction vs Real for Velocity {velocity} km/h')
        plt.legend()
        plt.grid(True)

        # Convert plot to an image for display in Streamlit
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return buf

    # Streamlit UI for Acceleration Predictions
    def acceleration_prediction(self):
        st.write("Use the form below to predict acceleration and view the results.")

        # User inputs
        start_cycle = st.number_input("Start Cycle:", min_value=0, value=0, step=50)
        end_cycle = st.number_input("End Cycle:", min_value=1000, value=50000, step=50)
        step_size = st.number_input("Step Size:", min_value=100, value=100, step=50)
        velocity = st.selectbox("Velocity (km/h):", options=[160, 210, 270, 320])


        if st.button("Generate Predictions"):
            if start_cycle < end_cycle:
                # Generate and display the plot
                plot_image = self.generate_and_plot_accelerations(
                    start_cycle=start_cycle,
                    end_cycle=end_cycle,
                    step_size=step_size,
                    velocity=velocity
                )
                return plot_image
                st.image(plot_image, caption="Acceleration Predictions", use_container_width=True)
            else:
                st.error("Start Cycle must be less than End Cycle.")
        return None
                
















    def _generate_and_plot_acceleration_multi_aligned(self, velocity_ranges, step_size):
        """
        Generates and plots multi-aligned acceleration predictions.

        Parameters:
            model_neg: Trained machine learning model for negative acceleration.
            model_pos: Trained machine learning model for positive acceleration.
            velocity_ranges: List of velocity ranges as [[velocity, start_cycle, end_cycle], ...].
            step_size: Step size for cycle generation.

        Returns:
            BytesIO: Buffer containing the plot image.
        """
        previous_neg_end_value = None
        previous_pos_end_value = None

        model_neg = self.n_max_rf_model
        model_pos = self.p_max_rf_model

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
                'Cycle_Number': cycle_numbers
            })

            # Predict negative and positive max accelerations
            predicted_neg_acceleration = model_neg.predict(new_data)
            predicted_pos_acceleration = model_pos.predict(new_data)

            # Adjust negative predictions for continuity
            if previous_neg_end_value is not None:
                adjustment = previous_neg_end_value - predicted_neg_acceleration[0]
                predicted_neg_acceleration += adjustment

            # Adjust positive predictions for continuity
            if previous_pos_end_value is not None:
                adjustment = previous_pos_end_value - predicted_pos_acceleration[0]
                predicted_pos_acceleration += adjustment

            # Update the end values for the next velocity
            previous_neg_end_value = predicted_neg_acceleration[-1]
            previous_pos_end_value = predicted_pos_acceleration[-1]

            # Plot negative predictions
            plt.plot(cycle_numbers, predicted_neg_acceleration,
                    label=f'Predicted Negative Max Acceleration (Velocity {velocity} km/h)',
                    color=color_map[velocity], linestyle='-')

            # Plot positive predictions
            plt.plot(cycle_numbers, predicted_pos_acceleration,
                    label=f'Predicted Positive Max Acceleration (Velocity {velocity} km/h)',
                    color=color_map[velocity], linestyle='--')

        # Plot settings
        plt.xlabel('Cycle Number')
        plt.ylabel('Acceleration (m/s²)')
        plt.title('Aligned Negative and Positive Max Acceleration Predictions')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Save plot to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return buf



    def acceleration_multi_aligned(self, cell_style):
        left, right = st.columns(2)
        with left:
            st.subheader("Acceleration Prediction")

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
                            <div style="border-bottom: 1px solid #d3d3d3; margin-bottom: 10px; padding: 0px;">
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

                        st.markdown(
                            f"""
                            <div style="border-bottom: 1px solid #d3d3d3; margin-bottom: 10px; padding: 0px;">
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

        if st.button("Generate Predictions", key="generate_predictions"):
            plot_buffer = self._generate_and_plot_acceleration_multi_aligned(velocity_ranges=velocity_ranges, step_size=step_size)
            st.image(plot_buffer, caption='Acceleration Predictions', use_container_width=True)

            # Generate and display table
            combined_results = []
            for velocity, start_cycle, end_cycle in velocity_ranges:
                cycle_numbers = list(range(start_cycle, end_cycle + 1, step_size))
                new_data = pd.DataFrame({'velocity': [velocity] * len(cycle_numbers), 'Cycle_Number': cycle_numbers})
                predicted_pos_acceleration = self.p_max_rf_model.predict(new_data)
                predicted_neg_acceleration = self.n_max_rf_model.predict(new_data)
                combined_results.extend(zip(new_data['velocity'], new_data['Cycle_Number'], predicted_pos_acceleration, predicted_neg_acceleration))

            result_df = pd.DataFrame(combined_results, columns=["velocity", "Cycle_Number", "smoothed_positive_max", "smoothed_negative_max"])
            st.dataframe(result_df)











