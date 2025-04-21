import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt


def interpolate_data_same_cam(filename1, filename2, time1, time2, angle1, angle2):
    # Read the data from the files
    data1 = pd.read_csv(filename1)
    data2 = pd.read_csv(filename2)

    # Initialize lists to store results
    interpolated_times = []
    resultant_angles = []

    # Iterate line by line through both datasets
    for (_, row1), (_, row2) in zip(data1.iterrows(), data2.iterrows()):
        time1_value = row1[time1]
        time2_value = row2[time2]
        angle1_value = row1[angle1]
        angle2_value = row2[angle2]

        # Handle missing angles
        if pd.isna(angle1_value) and pd.isna(angle2_value):
            resultant_angle = np.nan
        elif pd.isna(angle1_value):
            resultant_angle = angle2_value
        elif pd.isna(angle2_value):
            resultant_angle = angle1_value
        else:
            # Calculate the resultant angle
            angle1_rad = np.radians(angle1_value)
            angle2_rad = np.radians(angle2_value)
            resultant_angle = np.degrees(np.arctan2(
                np.sqrt(np.sin(angle1_rad) ** 2 + np.sin(angle2_rad) ** 2),
                np.cos(angle1_rad) * np.cos(angle2_rad)
            ))

        # Store the results
        interpolated_times.append(time1_value)
        resultant_angles.append(resultant_angle)

    # Create a DataFrame for the interpolated data
    interpolated_data = pd.DataFrame({
        'UTC Time': interpolated_times,
        'Resultant Angle': resultant_angles
    })

    # Save the interpolated data to a CSV file
    output_filename = 'interpolated_data_same_cam.csv'
    interpolated_data.to_csv(output_filename, index=False)
    print(f"Interpolated data saved to {output_filename}")

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(interpolated_data['UTC Time'], interpolated_data['Resultant Angle'], label='Resultant Angle')
    plt.xlabel('UTC Time')
    plt.ylabel('Resultant Angle (degrees)')
    plt.title('Interpolated Resultant Angles (Same Camera)')
    plt.legend()
    plt.grid()
    plt.show()


filename1 = '../raw_data/angle_data_1.5A.csv'
filename2 = '../raw_data/90angle_data_1.5A.csv'
# Extract the time and angle column names from the filenames
time1 = 'UTC Time'
angle1 = 'Angle'
time2 = 'UTC Time'
angle2 = 'Angle'
# Call the function to interpolate data
interpolate_data_same_cam(filename1, filename2, time1, time2, angle1, angle2)
