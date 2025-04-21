import pandas as pd
from datetime import datetime, timedelta


def fix_utc_time(file_paths, output_dir):
    """
    Fixes the UTC Time column in the given files by recalculating it using the Time column.

    Parameters:
    file_paths (list): List of file paths to process.
    output_dir (str): Directory to save the updated files.
    """
    for file_path in file_paths:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Parse the last UTC Time value
        last_utc_time = datetime.fromisoformat(df['UTC Time'].iloc[-1])

        # Get the final time elapsed
        final_time_elapsed = df['Time'].iloc[-1]

        # Update the UTC Time column with the desired format
        df['UTC Time'] = df['Time'].apply(
            lambda t: (last_utc_time - timedelta(seconds=(final_time_elapsed - t))).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
        )

        # Save the updated file
        output_path = f"{output_dir}/{file_path.split('/')[-1]}"
        df.to_csv(output_path, index=False)
        print(f"Updated file saved to: {output_path}")


# List of files to process
file_paths = [
    '/Users/joehavens/Documents/Documents/pythonProject/HelloWorld/PHSX 616/Magnetic_torque/raw_data/angle_data_0.5A.csv',
    '/Users/joehavens/Documents/Documents/pythonProject/HelloWorld/PHSX 616/Magnetic_torque/raw_data/angle_data_1.0A.csv',
    '/Users/joehavens/Documents/Documents/pythonProject/HelloWorld/PHSX 616/Magnetic_torque/raw_data/angle_data_1.5A.csv'
]

# Output directory for updated files
output_dir = '/Users/joehavens/Documents/Documents/pythonProject/HelloWorld/PHSX 616/Magnetic_torque/updated_data'

# Fix the UTC Time column in the files
fix_utc_time(file_paths, output_dir)
