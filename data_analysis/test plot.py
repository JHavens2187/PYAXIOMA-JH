import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

files = ['../raw_data/angle_data_0.5A.csv', '../raw_data/angle_data_1.0A.csv', '../raw_data/angle_data_1.5A.csv',
         '../raw_data/angle_data_2.0A.csv', '../raw_data/angle_data_3.5A.csv']

for file in files:
    with open(file, 'r') as f:
        data = pd.read_csv(f)
        time = data['Time']
        angle = data['Angle']
        # replace all 0.0 confidence values with nan then multiply each confidence value by 100
        confidence = data['Confidence'].replace(0.0, np.nan) * 100

        amps = float((file.split('_')[3].split('.')[0] + '.' + file.split('_')[3].split('.')[1]).replace('A', ''))

        '''
        # manually move the confidence values to the correct range
        if amps == 3.5:
            confidence -= 11 * amps
        elif amps == 2.0:
            confidence -= 10 * amps
        elif amps == 1.5:
            confidence += 3 * amps
        elif amps == 1.0:
            confidence += 8 * amps
        elif amps == 0.5:
            confidence += 40 * amps

        confidence = confidence.clip(lower=0, upper=100)'''

        raw_confidence = confidence
        angle_smooth = angle.rolling(window=20, min_periods=1).mean()
        angle_stability = 1 - np.clip(np.abs(angle - angle_smooth) / 30, 0, 1)

        # Compute angular velocity and jerk
        angle_np = angle.to_numpy()
        dtheta = np.gradient(angle_np)
        dt = np.gradient(time)
        angular_velocity = np.abs(dtheta / dt)
        jerk = np.gradient(angular_velocity)
        jerk_penalty = np.exp(- (np.abs(jerk) / 20.0)**2)

        # Angular velocity variance penalty (higher variance = less confidence)
        angular_var = pd.Series(angular_velocity).rolling(window=30, min_periods=1).std()
        angular_var_penalty = 1 - np.clip(angular_var / 10.0, 0, 1)

        # Penalize when angle is close to 90 deg (parallel motion)
        angle_parallel_penalty = np.exp(- ((angle - 90) / 25.0) ** 2)

        # Combine all modifiers (scale each to 0–100 range)
        modifier = (
            30 * angle_stability +
            25 * jerk_penalty +
            20 * angular_var_penalty +
            15 * angle_parallel_penalty
        )

        # Final confidence estimate
        confidence = (0.5 * raw_confidence.fillna(50) + 0.8 * (100 - modifier)).clip(0, 100)
        #confidence = 100 * (1 / (1 + np.exp(-((confidence - 50) / 10))))

        inv_confidence = 100 - confidence

        # add a moving average to the confidence values
        moving_confidence_avg = confidence.rolling(window=50, min_periods=1).mean()

    plt.figure(figsize=(10, 6))
    plt.scatter(time, angle, label='Angle', color='r', s=0.2)
    plt.scatter(time, confidence, label='Confidence (%)', color='b', s=0.1)
    plt.plot(time, moving_confidence_avg, label='Moving Average Confidence (%)', color='g', linewidth=1)
    plt.xlabel('Elapsed Time (s)')
    plt.ylabel('Resultant Angle (degrees) / Confidence (%)')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.title(f'Interpolated Resultant Angles {amps}A')
    plt.show()
