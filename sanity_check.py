import pandas as pd
import numpy as np


def check_flow_direction(df, angle_threshold=20):
    """
    Analyze flow direction consistency by comparing vector angles to the median flow direction.

    Calculates the angle of each velocity vector and determines what percentage of
    vectors align within a specified angular threshold of the median flow direction.
    Identifies outlier vectors that deviate significantly from the predominant flow.

    Args:
        df (pandas.DataFrame): DataFrame containing velocity data with 'u_velocity'
            and 'v_velocity' columns representing horizontal and vertical velocity
            components.
        angle_threshold (float, optional): Maximum angular deviation (in degrees) from
            the median flow direction for a vector to be considered consistent.
            Defaults to 20.

    Returns:
        tuple ([float, pandas.DataFrame]): A tuple containing:
            - consistent_flow (float): Percentage of vectors within the angle threshold
            of the median flow direction (0-100).
            - outliers (pandas.DataFrame): Subset of the input DataFrame containing only
            vectors that exceed the angle threshold.

    Notes:
        - Adds 'angle' and 'angle_diff' columns to the input DataFrame
        - Flow angles are calculated using arctan2(v_velocity, u_velocity)
        - Angular differences account for circular nature of angles (wraps at 360°)
    """

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
    """
    Analyze velocity distribution across the flow field by comparing center and edge regions.

    Divides the flow field into three horizontal regions (left edge, center, right edge)
    and calculates average velocity magnitudes for the center versus the edges. This
    helps identify boundary layer effects or non-uniform flow patterns.

    Args:
        df (pandas.DataFrame): DataFrame containing flow data with 'x_position' and
            'velocity_magnitude' columns.
        edge_zone_percent (float, optional): Fraction of the total horizontal range
            to designate as edge zones on each side. For example, 0.2 means the left
            20% and right 20% are edge zones, with the middle 60% as center.
            Defaults to 0.2.

    Returns:
        tuple ([float, float]): A tuple containing:
            - avg_center (float): Average velocity magnitude in the center region.
            - avg_edges (float): Average velocity magnitude in the combined edge regions
            (left and right).

    Notes:
        - Adds a 'region' column to the input DataFrame with values 'edge_left',
        'center', or 'edge_right'
        - Edge zones are symmetric on both sides of the flow field
        - Useful for detecting boundary layer effects or flow uniformity issues
    """
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
    """
    Identify velocity vectors that deviate significantly from their spatial neighbors.

    Examines each vector and compares its velocity magnitude to the average velocity
    of nearby vectors. Flags vectors as outliers if they differ from their neighbors
    by more than a specified threshold, indicating potential measurement errors or
    turbulent regions.

    Args:
        df (pandas.DataFrame): DataFrame containing flow data with 'x_position',
            'y_position', and 'velocity_magnitude' columns.
        distance_threshold (float, optional): Maximum distance (in position units)
            to consider other vectors as neighbors. Defaults to 20.
        velocity_diff_threshold (float, optional): Minimum velocity magnitude difference
            from the neighborhood average for a vector to be classified as an outlier.
            Defaults to 0.2.

    Returns:
        pandas.DataFrame: DataFrame containing outlier information with columns:
            - index (int): Original index of the outlier vector in the input DataFrame
            - x (float): X-position of the outlier
            - y (float): Y-position of the outlier
            - velocity (float): Velocity magnitude of the outlier
            - neighbor_avg_velocity (float): Average velocity of nearby neighbors
            - difference (float): Absolute velocity difference from neighbor average

    Notes:
        - Uses Euclidean distance to identify neighbors
        - Excludes the vector itself when calculating neighborhood average
        - Returns empty DataFrame if no outliers are found
        - Useful for quality control and identifying spurious velocity measurements
    """
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


