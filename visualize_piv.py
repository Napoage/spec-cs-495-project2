import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys
sys.path.append("PIV")
from ensemble_PIV import ensemble_piv

def run_piv_with_visualization():
    """Run PIV analysis and create visualization of results"""

    print("Running PIV analysis with visualization...")

    # Get image files
    image_dir = "images"
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

    if len(image_files) < 2:
        print("Need at least 2 images for PIV analysis")
        return

    # Load first two images
    img1_path = os.path.join(image_dir, image_files[0])
    img2_path = os.path.join(image_dir, image_files[1])

    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    print(f"Processing images: {image_files[0]} and {image_files[1]}")
    print(f"Image dimensions: {img1.shape}")

    # Simple PIV parameters
    piv_params = {
        'int_area': 64,  # Interrogation window size
        'step': 32,      # Step size (overlap)
        'subpixel_method': 'gaussian',
        'mask_thresh': 0.05,
        'filter_method': 'localmean',
        'max_filter_iteration': 10,
        'filter_kernel': 2,
        'infillFlag': True,
        'smoothFlag': True
    }

    # Create image stack
    stack = {
        'pre_proc': [img1_path, img2_path],
        'final_roi': np.ones_like(img1, dtype=bool),  # Process entire image
        'Rcrop': []
    }

    try:
        # Run PIV - simplified call
        print("Running ensemble PIV...")

        # Process images directly
        images = [img1.astype(np.float64), img2.astype(np.float64)]

        # Set up PIV grid
        int_area = piv_params['int_area']
        step = piv_params['step']

        # Create coordinate meshgrid
        y_coords = range(int_area//2, img1.shape[0] - int_area//2, step)
        x_coords = range(int_area//2, img1.shape[1] - int_area//2, step)
        x_piv, y_piv = np.meshgrid(x_coords, y_coords)

        print(f"PIV grid size: {x_piv.shape}")

        # Simple cross-correlation PIV
        u_piv = np.zeros_like(x_piv, dtype=float)
        v_piv = np.zeros_like(y_piv, dtype=float)

        for i in range(x_piv.shape[0]):
            for j in range(x_piv.shape[1]):
                # Get interrogation windows
                x_center, y_center = int(x_piv[i,j]), int(y_piv[i,j])

                # Extract windows
                half_win = int_area // 2
                y1, y2 = y_center - half_win, y_center + half_win
                x1, x2 = x_center - half_win, x_center + half_win

                if y1 >= 0 and y2 < img1.shape[0] and x1 >= 0 and x2 < img1.shape[1]:
                    win1 = img1[y1:y2, x1:x2]
                    win2 = img2[y1:y2, x1:x2]

                    # Cross-correlation
                    corr = cv2.matchTemplate(win1, win2, cv2.TM_CCORR_NORMED)
                    _, _, _, max_loc = cv2.minMaxLoc(corr)

                    # Calculate displacement
                    dy, dx = max_loc[1] - corr.shape[0]//2, max_loc[0] - corr.shape[1]//2
                    u_piv[i,j] = dx
                    v_piv[i,j] = dy

        # Create visualization
        create_piv_visualization(img1, x_piv, y_piv, u_piv, v_piv,
                               image_files[0], image_files[1])

        print("PIV visualization complete!")

    except Exception as e:
        print(f"Error in PIV processing: {e}")
        import traceback
        traceback.print_exc()

def create_piv_visualization(background_img, x_piv, y_piv, u_piv, v_piv,
                           img1_name, img2_name):
    """Create and save PIV visualization"""

    plt.figure(figsize=(15, 10))

    # Plot 1: Background image with velocity vectors
    plt.subplot(2, 2, 1)
    plt.imshow(background_img, cmap='gray')

    # Subsample vectors for cleaner visualization
    skip = max(1, len(x_piv[0]) // 20)  # Show every nth vector

    plt.quiver(x_piv[::skip, ::skip], y_piv[::skip, ::skip],
              u_piv[::skip, ::skip], v_piv[::skip, ::skip],
              scale=50, color='red', alpha=0.8, width=0.003)

    plt.title(f'PIV Velocity Vectors\n{img1_name} → {img2_name}')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')

    # Plot 2: U velocity component
    plt.subplot(2, 2, 2)
    plt.imshow(u_piv, cmap='coolwarm', origin='lower')
    plt.colorbar(label='U velocity (pixels/frame)')
    plt.title('U Velocity Component (horizontal)')

    # Plot 3: V velocity component
    plt.subplot(2, 2, 3)
    plt.imshow(v_piv, cmap='coolwarm', origin='lower')
    plt.colorbar(label='V velocity (pixels/frame)')
    plt.title('V Velocity Component (vertical)')

    # Plot 4: Velocity magnitude
    plt.subplot(2, 2, 4)
    magnitude = np.sqrt(u_piv**2 + v_piv**2)
    plt.imshow(magnitude, cmap='viridis', origin='lower')
    plt.colorbar(label='Velocity magnitude (pixels/frame)')
    plt.title('Velocity Magnitude')

    plt.tight_layout()
    plt.savefig('piv_results.png', dpi=150, bbox_inches='tight')
    plt.savefig('piv_results.pdf', bbox_inches='tight')

    print(f"Visualization saved as 'piv_results.png' and 'piv_results.pdf'")

    # Print statistics
    print(f"\nVelocity Statistics:")
    print(f"U velocity - Mean: {np.nanmean(u_piv):.2f}, Std: {np.nanstd(u_piv):.2f}")
    print(f"V velocity - Mean: {np.nanmean(v_piv):.2f}, Std: {np.nanstd(v_piv):.2f}")
    print(f"Magnitude - Mean: {np.nanmean(magnitude):.2f}, Max: {np.nanmax(magnitude):.2f}")

    plt.show()

if __name__ == "__main__":
    run_piv_with_visualization()