import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os
import sys

# Add PIV directory to path
sys.path.append("PIV")
from ensemble_PIV import ensemble_piv

def extract_and_save_piv_results():
    """Extract PIV results and save them manually"""

    print("Extracting PIV results from processed data...")

    # Get image files
    image_dir = "images"
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

    if len(image_files) < 2:
        print("Need at least 2 images")
        return

    print(f"Found {len(image_files)} images")

    # Simple PIV parameters matching the SPEC system
    piv_params = {
        'int_area': 68,  # From the log output
        'step': 32,
        'subpixel_method': 'gaussian',
        'infillFlag': True,
        'smoothFlag': True,
        'filter_method': 'localmean',
        'max_filter_iteration': 10,
        'filter_kernel': 2
    }

    # Load first two images to get dimensions
    img1 = cv2.imread(os.path.join(image_dir, image_files[0]), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(os.path.join(image_dir, image_files[1]), cv2.IMREAD_GRAYSCALE)

    print(f"Image dimensions: {img1.shape}")

    # Create a simple mask (no masking - process entire image)
    mask = np.ones(img1.shape, dtype=bool)

    # Stack all images for ensemble PIV
    image_stack = []
    for img_file in image_files:
        img = cv2.imread(os.path.join(image_dir, img_file), cv2.IMREAD_GRAYSCALE)
        image_stack.append(img.astype(np.float64))

    print(f"Processing {len(image_stack)} images with ensemble PIV...")

    # Run ensemble PIV (this is the core of what SPEC does)
    try:
        BASE_DIR = "."
        results = ensemble_piv(
            image_stack,
            piv_params['int_area'],
            piv_params['step'],
            piv_params['subpixel_method'],
            mask,
            piv_params['infillFlag'],
            piv_params['smoothFlag'],
            piv_params['filter_method'],
            piv_params['max_filter_iteration'],
            piv_params['filter_kernel'],
            BASE_DIR
        )

        # Extract results
        x_piv, y_piv, u_piv, v_piv, type_vec, corr_map = results

        print(f"PIV grid shape: {x_piv.shape}")
        print(f"Valid vectors: {np.sum(~np.isnan(u_piv))}/{u_piv.size}")

        # Save results to CSV
        save_piv_to_csv(x_piv, y_piv, u_piv, v_piv, type_vec, corr_map)

        # Create visualization
        create_piv_visualization(img1, x_piv, y_piv, u_piv, v_piv)

        print("PIV results extracted and saved successfully!")

    except Exception as e:
        print(f"Error in ensemble PIV: {e}")
        import traceback
        traceback.print_exc()

def save_piv_to_csv(x_piv, y_piv, u_piv, v_piv, type_vec, corr_map):
    """Save PIV results to CSV files with proper formatting"""

    # Create results directory
    os.makedirs("piv_csv_results", exist_ok=True)
    print(f"Created directory: piv_csv_results")

    # Save individual component matrices (grid format)
    components = {
        'x_positions': x_piv,
        'y_positions': y_piv,
        'u_velocity': u_piv,
        'v_velocity': v_piv,
        'velocity_magnitude': np.sqrt(u_piv**2 + v_piv**2),
        'correlation': corr_map,
        'vector_type': type_vec
    }

    print(f"Grid shape: {x_piv.shape}")

    # Save each component as a matrix (preserving spatial structure)
    for name, data in components.items():
        csv_path = f"piv_csv_results/{name}.csv"
        # Use pandas to save with proper float formatting
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False, header=False, float_format='%.6f')
        print(f"✓ Saved {name} ({data.shape}) to {csv_path}")

    # Save point-by-point data (each row = one measurement point)
    combined_data = []
    valid_count = 0

    print("Processing grid points...")
    for i in range(x_piv.shape[0]):
        for j in range(x_piv.shape[1]):
            x_val = x_piv[i,j]
            y_val = y_piv[i,j]
            u_val = u_piv[i,j]
            v_val = v_piv[i,j]
            mag_val = np.sqrt(u_val**2 + v_val**2)
            corr_val = corr_map[i,j]
            type_val = type_vec[i,j]

            combined_data.append([
                i, j,                    # Grid indices
                x_val, y_val,            # Physical positions
                u_val, v_val, mag_val,   # Velocities
                corr_val, type_val       # Quality metrics
            ])

            # Count valid (non-NaN) vectors
            if not (np.isnan(u_val) or np.isnan(v_val)):
                valid_count += 1

    # Create DataFrame with proper column names
    combined_df = pd.DataFrame(combined_data, columns=[
        'grid_i', 'grid_j',           # Grid indices (row, col)
        'x_pixel', 'y_pixel',         # Position in pixels
        'u_velocity', 'v_velocity', 'velocity_magnitude',  # Velocity components
        'correlation', 'vector_type'   # Quality indicators
    ])

    # Save combined results
    combined_path = "piv_csv_results/combined_results.csv"
    combined_df.to_csv(combined_path, index=False, float_format='%.6f')
    print(f"✓ Saved combined point data ({len(combined_data)} points) to {combined_path}")

    # Save only valid vectors (non-NaN)
    valid_df = combined_df.dropna(subset=['u_velocity', 'v_velocity'])
    valid_path = "piv_csv_results/valid_vectors_only.csv"
    valid_df.to_csv(valid_path, index=False, float_format='%.6f')
    print(f"✓ Saved valid vectors only ({len(valid_df)} points) to {valid_path}")

    # Create summary statistics file
    stats_data = {
        'Parameter': ['Grid_Rows', 'Grid_Cols', 'Total_Points', 'Valid_Points', 'Valid_Percentage',
                     'U_Mean', 'U_Std', 'U_Min', 'U_Max',
                     'V_Mean', 'V_Std', 'V_Min', 'V_Max',
                     'Speed_Mean', 'Speed_Max', 'Correlation_Mean'],
        'Value': [x_piv.shape[0], x_piv.shape[1], x_piv.size, valid_count, 100*valid_count/x_piv.size,
                 np.nanmean(u_piv), np.nanstd(u_piv), np.nanmin(u_piv), np.nanmax(u_piv),
                 np.nanmean(v_piv), np.nanstd(v_piv), np.nanmin(v_piv), np.nanmax(v_piv),
                 np.nanmean(np.sqrt(u_piv**2 + v_piv**2)), np.nanmax(np.sqrt(u_piv**2 + v_piv**2)),
                 np.nanmean(corr_map)]
    }

    stats_df = pd.DataFrame(stats_data)
    stats_path = "piv_csv_results/statistics.csv"
    stats_df.to_csv(stats_path, index=False, float_format='%.6f')
    print(f"✓ Saved statistics summary to {stats_path}")

    print(f"\n📊 CSV Export Summary:")
    print(f"   • Grid format: {len(components)} matrix files")
    print(f"   • Point format: 1 combined file with {len(combined_data)} rows")
    print(f"   • Valid vectors: {valid_count}/{x_piv.size} ({100*valid_count/x_piv.size:.1f}%)")
    print(f"   • Location: piv_csv_results/ directory")

def create_piv_visualization(background_img, x_piv, y_piv, u_piv, v_piv):
    """Create PIV visualization"""

    plt.figure(figsize=(16, 12))

    # Plot 1: Background with vectors
    plt.subplot(2, 3, 1)
    plt.imshow(background_img, cmap='gray')

    # Subsample for cleaner arrows
    skip = max(1, x_piv.shape[1] // 20)
    scale = np.nanmax(np.sqrt(u_piv**2 + v_piv**2)) * 3

    plt.quiver(x_piv[::skip, ::skip], y_piv[::skip, ::skip],
              u_piv[::skip, ::skip], v_piv[::skip, ::skip],
              scale=scale, color='red', alpha=0.8, width=0.002)
    plt.title('Water Surface with Velocity Vectors')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')

    # Plot 2: U velocity
    plt.subplot(2, 3, 2)
    u_masked = np.ma.masked_invalid(u_piv)
    im1 = plt.imshow(u_masked, cmap='RdBu_r', origin='lower')
    plt.colorbar(im1, label='U velocity (pixels/frame)')
    plt.title('Horizontal Velocity (U)')

    # Plot 3: V velocity
    plt.subplot(2, 3, 3)
    v_masked = np.ma.masked_invalid(v_piv)
    im2 = plt.imshow(v_masked, cmap='RdBu_r', origin='lower')
    plt.colorbar(im2, label='V velocity (pixels/frame)')
    plt.title('Vertical Velocity (V)')

    # Plot 4: Speed magnitude
    plt.subplot(2, 3, 4)
    magnitude = np.sqrt(u_piv**2 + v_piv**2)
    mag_masked = np.ma.masked_invalid(magnitude)
    im3 = plt.imshow(mag_masked, cmap='viridis', origin='lower')
    plt.colorbar(im3, label='Speed (pixels/frame)')
    plt.title('Velocity Magnitude')

    # Plot 5: Vector field streamlines
    plt.subplot(2, 3, 5)
    x_stream = np.arange(0, x_piv.shape[1])
    y_stream = np.arange(0, x_piv.shape[0])
    X_stream, Y_stream = np.meshgrid(x_stream, y_stream)

    # Interpolate velocity to regular grid for streamlines
    u_interp = np.nan_to_num(u_piv)
    v_interp = np.nan_to_num(v_piv)

    plt.streamplot(X_stream, Y_stream, u_interp, v_interp,
                  density=1, color='blue', linewidth=1)
    plt.title('Flow Streamlines')
    plt.xlabel('Grid X')
    plt.ylabel('Grid Y')

    # Plot 6: Statistics summary
    plt.subplot(2, 3, 6)
    plt.axis('off')

    # Calculate statistics
    valid_u = u_piv[~np.isnan(u_piv)]
    valid_v = v_piv[~np.isnan(v_piv)]
    valid_mag = magnitude[~np.isnan(magnitude)]

    stats_text = f"""PIV Results Summary

Grid Size: {x_piv.shape[0]} × {x_piv.shape[1]} = {x_piv.size} points
Valid Vectors: {len(valid_u)}/{x_piv.size} ({100*len(valid_u)/x_piv.size:.1f}%)

Horizontal Velocity (U):
  Mean: {np.mean(valid_u):.3f} pixels/frame
  Std:  {np.std(valid_u):.3f} pixels/frame
  Range: [{np.min(valid_u):.2f}, {np.max(valid_u):.2f}]

Vertical Velocity (V):
  Mean: {np.mean(valid_v):.3f} pixels/frame
  Std:  {np.std(valid_v):.3f} pixels/frame
  Range: [{np.min(valid_v):.2f}, {np.max(valid_v):.2f}]

Speed:
  Mean: {np.mean(valid_mag):.3f} pixels/frame
  Max:  {np.max(valid_mag):.3f} pixels/frame"""

    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig('final_piv_results.png', dpi=150, bbox_inches='tight')
    plt.savefig('final_piv_results.pdf', bbox_inches='tight')

    print("Final PIV visualization saved as 'final_piv_results.png'")
    plt.show()

if __name__ == "__main__":
    extract_and_save_piv_results()