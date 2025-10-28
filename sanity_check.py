import pandas as pd
import numpy as np

#import csv
df = pd.read_csv('piv_results.csv')

#calculate flow angles (degrees)
df['angle'] = np.degrees(np.arctan2(df['v_velocity'], df['u_velocity']))

#get median angle
median_angle = df['angle'].median()

#calcualte angular difference from median
def angular_difference(angle1, angle2):
    diff = (angle1 - angle2 + 180) % 360 - 180
    return np.abs(diff)

#calcuate angular difference for each vector
df['angle_diff'] = df['angle'].apply(lambda x: angular_difference(x, median_angle))

#threshold for consistent flow direction currently 30 degrees
consistent_flow = (df['angle_diff'] < 30).sum() / len(df) * 100
print(f"Percentage of vectors within 30° of median flow direction ({median_angle:.2f}°): {consistent_flow:.2f}%")

#find number of outliers
outliers = df[df['angle_diff'] > 30]
print(f"Nuber of directional outliers: {len(outliers)}")