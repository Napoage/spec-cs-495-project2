import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def visualize_piv_csv():
    """Visualize PIV results from CSV file"""

    # Load the PIV results
    print("Loading PIV results from CSV...")
    df = pd.read_csv('piv_results.csv')

    print(f"Loaded {len(df)} velocity vectors")
    print(f"Columns: {list(df.columns)}")

    # Basic statistics
    print(f"\nVelocity Statistics:")
    print(f"U-velocity: {df['u_velocity'].mean():.3f} ± {df['u_velocity'].std():.3f} m/s")
    print(f"V-velocity: {df['v_velocity'].mean():.3f} ± {df['v_velocity'].std():.3f} m/s")
    print(f"Speed: {df['velocity_magnitude'].mean():.3f} m/s (max: {df['velocity_magnitude'].max():.3f})")
    print(f"Correlation: {df['correlation'].mean():.3f}")

    # Load background image
    image_files = sorted([f for f in os.listdir('images') if f.endswith('.png')])
    if image_files:
        bg_img = cv2.imread(f'images/{image_files[0]}', cv2.IMREAD_GRAYSCALE)
        print(f"Background image: {image_files[0]} ({bg_img.shape})")
    else:
        bg_img = None
        print("No background image found")

    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 15))

    # Plot 1: Vector field on background
    plt.subplot(2, 3, 1)
    if bg_img is not None:
        plt.imshow(bg_img, cmap='gray', alpha=0.8)

    # Subsample vectors for cleaner display
    step = max(1, len(df) // 200)  # Show ~200 vectors max
    subset = df.iloc[::step]

    # Scale arrows
    scale = df['velocity_magnitude'].quantile(0.95) * 20

    # Fix coordinate system - flip Y-axis for correct flow direction display
    plt.quiver(subset['x_position'], subset['y_position'],
              subset['u_velocity'], -subset['v_velocity'],  # Flip V to match image coords
              scale=scale, color='red', alpha=0.8, width=0.002)

    plt.title('PIV Vector Field\n(Red arrows show flow direction & speed)')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')

    # Plot 2: U velocity heatmap
    plt.subplot(2, 3, 2)

    # Create grid for heatmap
    x_unique = sorted(df['x_position'].unique())
    y_unique = sorted(df['y_position'].unique())

    if len(x_unique) > 1 and len(y_unique) > 1:
        u_grid = np.full((len(y_unique), len(x_unique)), np.nan)

        for _, row in df.iterrows():
            try:
                i = y_unique.index(row['y_position'])
                j = x_unique.index(row['x_position'])
                u_grid[i, j] = row['u_velocity']
            except ValueError:
                continue

        im1 = plt.imshow(u_grid, cmap='RdBu_r', origin='lower', aspect='auto')
        plt.colorbar(im1, label='U velocity (m/s)')
        plt.title('Horizontal Velocity (U)')
        plt.xlabel('Grid X')
        plt.ylabel('Grid Y')
    else:
        plt.scatter(df['x_position'], df['y_position'], c=df['u_velocity'],
                   cmap='RdBu_r', s=20)
        plt.colorbar(label='U velocity (m/s)')
        plt.title('Horizontal Velocity (U)')

    # Plot 3: V velocity heatmap
    plt.subplot(2, 3, 3)
    if len(x_unique) > 1 and len(y_unique) > 1:
        v_grid = np.full((len(y_unique), len(x_unique)), np.nan)

        for _, row in df.iterrows():
            try:
                i = y_unique.index(row['y_position'])
                j = x_unique.index(row['x_position'])
                v_grid[i, j] = row['v_velocity']
            except ValueError:
                continue

        # Flip V-velocity display to match coordinate system
        im2 = plt.imshow(-v_grid, cmap='RdBu_r', origin='lower', aspect='auto')
        plt.colorbar(im2, label='V velocity (m/s, corrected)')
        plt.title('Vertical Velocity (V) - Corrected')
        plt.xlabel('Grid X')
        plt.ylabel('Grid Y')
    else:
        plt.scatter(df['x_position'], df['y_position'], c=-df['v_velocity'],
                   cmap='RdBu_r', s=20)
        plt.colorbar(label='V velocity (m/s, corrected)')
        plt.title('Vertical Velocity (V) - Corrected')

    # Plot 4: Velocity magnitude
    plt.subplot(2, 3, 4)
    if len(x_unique) > 1 and len(y_unique) > 1:
        mag_grid = np.full((len(y_unique), len(x_unique)), np.nan)

        for _, row in df.iterrows():
            try:
                i = y_unique.index(row['y_position'])
                j = x_unique.index(row['x_position'])
                mag_grid[i, j] = row['velocity_magnitude']
            except ValueError:
                continue

        im3 = plt.imshow(mag_grid, cmap='viridis', origin='lower', aspect='auto')
        plt.colorbar(im3, label='Speed (m/s)')
        plt.title('Velocity Magnitude')
        plt.xlabel('Grid X')
        plt.ylabel('Grid Y')
    else:
        plt.scatter(df['x_position'], df['y_position'], c=df['velocity_magnitude'],
                   cmap='viridis', s=20)
        plt.colorbar(label='Speed (m/s)')
        plt.title('Velocity Magnitude')

    # Plot 5: Histograms
    plt.subplot(2, 3, 5)
    plt.hist(df['u_velocity'], bins=30, alpha=0.7, label='U velocity', color='blue')
    plt.hist(df['v_velocity'], bins=30, alpha=0.7, label='V velocity', color='red')
    plt.hist(df['velocity_magnitude'], bins=30, alpha=0.7, label='Speed', color='green')
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Count')
    plt.title('Velocity Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 6: Quality metrics
    plt.subplot(2, 3, 6)

    # Correlation vs Speed scatter
    plt.scatter(df['correlation'], df['velocity_magnitude'], alpha=0.6, s=10)
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Velocity Magnitude (m/s)')
    plt.title('Measurement Quality vs Speed')
    plt.grid(True, alpha=0.3)

    # Add trend line
    valid_corr = df['correlation'].notna()
    if valid_corr.sum() > 1:
        z = np.polyfit(df[valid_corr]['correlation'],
                      df[valid_corr]['velocity_magnitude'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(df['correlation'].min(), df['correlation'].max(), 100)
        plt.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)

    plt.tight_layout()

    # Save visualization
    plt.savefig('piv_visualization.png', dpi=150, bbox_inches='tight')
    plt.savefig('piv_visualization.pdf', bbox_inches='tight')

    print(f"\n📊 Visualization saved as:")
    print(f"• piv_visualization.png")
    print(f"• piv_visualization.pdf")

    # Show additional analysis
    print(f"\n🔍 Data Analysis:")
    print(f"• Grid dimensions: {len(x_unique)} x {len(y_unique)} points")
    print(f"• Coverage area: {df['x_position'].max() - df['x_position'].min():.0f} x {df['y_position'].max() - df['y_position'].min():.0f} pixels")
    print(f"• Flow direction: {np.degrees(np.arctan2(df['v_velocity'].mean(), df['u_velocity'].mean())):.1f}° from horizontal")
    print(f"• High-speed areas: {(df['velocity_magnitude'] > df['velocity_magnitude'].quantile(0.8)).sum()} vectors")
    print(f"• Low correlation: {(df['correlation'] < 0.1).sum()} vectors ({100*(df['correlation'] < 0.1).sum()/len(df):.1f}%)")

    plt.show()
    return df

if __name__ == "__main__":
    visualize_piv_csv()