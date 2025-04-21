import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import glob

file_directory = '/Users/joehavens/Documents/Documents/pythonProject/HelloWorld/PHSX 616/Magnetic_torque/raw_data'


def interpolate_data(filename1, filename2, time1, time2, angle1, angle2, threshold=0.025):
    # Read the data from the files
    data1 = pd.read_csv(filename1)
    data2 = pd.read_csv(filename2)

    # Convert UTC time columns to datetime objects
    data1[time1] = pd.to_datetime(data1[time1])
    data2[time2] = pd.to_datetime(data2[time2])

    # Ensure data1[time1] is timezone-aware
    if data1[time1].dt.tz is None:
        data1[time1] = data1[time1].dt.tz_localize('UTC')
    # Ensure data2[time2] is timezone-aware
    if data2[time2].dt.tz is None:
        data2[time2] = data2[time2].dt.tz_localize('UTC')

    # Initialize lists to store results
    interpolated_times = []
    resultant_angles = []

    # Iterate through the first dataset
    for _, row1 in data1.iterrows():
        time1_value = row1[time1]
        angle1_value = row1[angle1]

        # Convert time1_value to timezone-aware (UTC)
        if time1_value.tzinfo is None:
            time1_value = time1_value.tz_localize('UTC')

        # Find the closest matching time in the second dataset within the threshold
        closest_row = data2.loc[
            (data2[time2] >= time1_value - timedelta(seconds=threshold)) &
            (data2[time2] <= time1_value + timedelta(seconds=threshold))
        ]

        if not closest_row.empty:
            # Use the first match (if multiple matches exist)
            time2_value = closest_row.iloc[0][time2]
            angle2_value = closest_row.iloc[0][angle2]

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

        if closest_row.empty:
            # If no match is found, append NaN
            interpolated_times.append(time1_value)
            resultant_angles.append(np.nan)

    # Create a DataFrame for the interpolated data
    interpolated_data = pd.DataFrame({
        'UTC Time': interpolated_times,
        'Resultant Angle': resultant_angles
    })

    # Save the interpolated data to a CSV file
    output_filename = 'interpolated_data.csv'
    interpolated_data.to_csv(output_filename, index=False)
    print(f"Interpolated data saved to {output_filename}")

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(interpolated_data['UTC Time'], interpolated_data['Resultant Angle'], label='Resultant Angle')
    plt.xlabel('UTC Time')
    plt.ylabel('Resultant Angle (degrees)')
    plt.title('Interpolated Resultant Angles')
    plt.legend()
    plt.grid()
    plt.show()


data1 = []
data2 = []

'''filename1 = '../raw_data/angle_data_1.5A.csv'
filename2 = '../raw_data/90angle_data_1.5A.csv'
# Extract the time and angle column names from the filenames
time1 = 'UTC Time'
angle1 = 'Angle'
time2 = 'UTC Time'
angle2 = 'Angle'
# Call the function to interpolate data
interpolate_data(filename1, filename2, time1, time2, angle1, angle2)'''

for filename in glob.glob(f"{file_directory}/*.csv"):
    if filename.startswith(f'{file_directory}/angle_data_'):
        data1.append(filename)
    #elif filename.startswith(f'{file_directory}/ball_angles_current_'):
    elif filename.startswith(f'{file_directory}/ball_angles_current_'):
        data2.append(filename)
    else: print("fuck")
    
for files in zip(data1, data2):
    filename1 = files[0]
    filename2 = files[1]

    # Extract the time and angle column names from the filenames
    time1 = 'UTC Time'
    angle1 = 'Angle'
    time2 = 'UTC Time'
    angle2 = 'Angle (degrees)'

    # Call the function to interpolate data
    interpolate_data(filename1, filename2, time1, time2, angle1, angle2)
