import pandas as pd
import numpy as np


def check_flow_direction(df, angle_threshold=20):

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
    consistent_flow = (df['angle_diff'] < angle_threshold).sum() / len(df) * 100
    #print(f"Percentage of vectors within {angle_threshold}° of median flow direction ({median_angle:.2f}°): {consistent_flow:.2f}%")

    #find number of outliers
    outliers = df[df['angle_diff'] > angle_threshold]
    #print(f"Nuber of directional outliers: {len(outliers)}")
    
    return consistent_flow, outliers


def check_velocity_profile(df, edge_zone_percent=.2):
    
    x_min = df['x_position'].min()
    x_max = df['x_position'].max()
    x_range = x_max - x_min

    #define left and right edge boundaries
    left_edge = x_min + edge_zone_percent * x_range
    right_edge = x_max - edge_zone_percent * x_range

    df['reigon'] = pd.cut(df['x_position'],#defines three reigons based on EDGE_ZONE_PERCENT
                        bins=[x_min, left_edge, right_edge, x_max],
                        labels=['edge_left', 'center', 'edge_right'])

    avg_center = df[df['reigon'] == 'center']['velocity_magnitude'].mean() #average velocity of center reigon
    avg_edges = df[df['reigon'].isin(['edge_left', 'edge_right'])]['velocity_magnitude'].mean() #average velocity of edge reigons
    
    return avg_center, avg_edges


def find_spatial_outliers(df, distance_threshold=20, velocity_diff_threshold=0.2):
    outliers = []

    for idx, row in df.iterrows():

        #calculate distance to all other points
        distances = np.sqrt(
            (df['x_position'] - row['x_position'])**2 +
            (df['y_position'] - row['y_position'])**2
        )

        neighbors = df[(distances > 0) & (distances < distance_threshold)]

        if len(neighbors) > 0:#if threre are neighbors within the distance threshold

            neighbor_avg_velocity = neighbors['velocity_magnitude'].mean() #calculate average velocity of neighbors
            velocity_difference = abs(row['velocity_magnitude'] - neighbor_avg_velocity)

            if velocity_difference > velocity_diff_threshold:
                outliers.append({
                    'index': idx,
                    'x': row['x_position'],
                    'y': row['y_position'],
                    'velocity': row['velocity_magnitude'],
                    'neighbor_avg_velocity': neighbor_avg_velocity,
                    'difference': velocity_difference
                })


    return pd.DataFrame(outliers)

#import csv
"""df = pd.read_csv('piv_results.csv')

spatial_outliers = find_spatial_outliers(df, distance_threshold=40, velocity_diff_threshold=.1)
print("**********Spatial Consistency Check**********")
print(f"Number of spatial outliers: {len(spatial_outliers)}")
print(f"Percentage of spatial outliers: {len(spatial_outliers)/len(df)*100:.1f}%")
print("")

consistent_flow, directional_outliers = check_flow_direction(df, angle_threshold=25)
print("**********Flow Direction Consistency Check**********")
print(f"Number of directional outliers: {len(directional_outliers)}")
print(f"Percentage of vectors within 25° of median flow direction: {consistent_flow:.2f}%")
print("")

avg_center_velocity, avg_edge_velocity = check_velocity_profile(df, edge_zone_percent=.2)
print("**********Velocity Profile Check**********")
print(f"Average center velocity: {avg_center_velocity:.3f}")
print(f"Average edge velocity: {avg_edge_velocity:.3f}")
print(f"Ratio (center/edge): {avg_center_velocity/avg_edge_velocity:.2f}")
if avg_center_velocity > avg_edge_velocity:
    print("Center is faster than edges (GOOD)")
else:
    print("Center is not faster than edges (BAD)")"""


