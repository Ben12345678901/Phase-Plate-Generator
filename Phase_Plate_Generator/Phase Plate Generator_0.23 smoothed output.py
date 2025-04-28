import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Polygon, Circle
import time
from scipy.interpolate import RectBivariateSpline

# Record the starting time to measure total execution time
start = time.time()

def load_input_beam(filepath, target_shape):
    """
    Load the input beam intensity profile or initial phase (e.g., from a wavefront sensor)
    from a NumPy array file.
    
    Parameters:
      filepath: Path to the NumPy file (e.g., 'input_beam.npy').
      target_shape: Desired shape (rows, cols) for the simulation grid.
      
    Returns:
      beam: 2D numpy array with normalized intensity values.
    """
    # Load the beam data from file
    beam = np.load(filepath)
    # If there are extra dimensions, remove them to obtain a 2D array
    if beam.ndim > 2:
        beam = np.squeeze(beam)
    
    # Convert to float64 for precision in subsequent computations
    beam = beam.astype(np.float64)
    
    # If the beam's shape doesn't match the target shape, resize it using OpenCV's interpolation
    if beam.shape != target_shape:
        import cv2  # cv2 is handy for resizing images/arrays
        beam = cv2.resize(beam, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
    
    return beam

def load_input_beam_png(filepath, target_shape):
    """
    Load the input beam intensity profile from an image file.
    
    Parameters:
      filepath: Path to the intensity image file.
      target_shape: Tuple specifying the desired shape (height, width).
      
    Returns:
      beam: 2D numpy array with normalized intensity values (ranging from 0 to 1).
    """
    # Read the image in grayscale mode
    beam = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if beam is None:
        raise ValueError(f"Could not read file: {filepath}")
    # Convert the image data to float64 (normalization can be applied if needed)
    beam = beam.astype(np.float64)
    
    # Resize the image if its shape does not match the target shape
    if beam.shape != target_shape:
        beam = cv2.resize(beam, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
    return beam

def compute_phase_elements(plate_size: float, wavelength: float, focal_length: float, desired_focal_spot: float):
    """
    Compute the phase element size and grid parameters based on the desired focal spot size.
    
    Parameters:
      plate_size: Physical size of the phase plate in meters.
      wavelength: Laser wavelength in meters.
      focal_length: Focal length of the optical system in meters.
      desired_focal_spot: Desired focal spot size in meters.
    
    Returns:
      element_size: The computed "diameter" (horizontal corner-to-corner distance) of each flat-top hexagon (m).
      num_elements: Approximate number of phase elements along one side.
      element_area: Area of each phase element (m²).
      focal_spot_size: Computed focal spot size (m) based on element_size.
    """
    # Compute element size using the formula: element_size = (wavelength * focal_length) / desired_focal_spot
    element_size = wavelength * focal_length / desired_focal_spot
    print(f"Computed element size: {element_size:.4e} m (from desired focal spot: {desired_focal_spot:.4e} m)")
    
    # Determine how many elements can fit along one side of the phase plate
    num_elements = int(plate_size / element_size)
    if num_elements < 8:
        raise ValueError(
            f"Computed number of phase elements ({num_elements}) is too low. "
            f"Consider increasing plate_size (currently {plate_size} m) or "
            f"increasing desired_focal_spot (currently {desired_focal_spot} m) "
            f"to obtain a higher resolution grid."
        )
    
    # Calculate the area of each phase element assuming a square (for simplicity)
    element_area = element_size ** 2
    print(f"Area of each phase element: {element_area:.4e} m² with {num_elements} elements along each side.")

    # Compute the focal spot size from the element size
    focal_spot_size = wavelength * focal_length / element_size
    print(f"Computed focal spot size (from element size): {focal_spot_size:.4e} m")
    
    return element_size, num_elements, element_area, focal_spot_size

def build_flat_top_hex_grid(plate_size: float, element_size: float):
    """
    Build a flat-topped hexagonal grid (with no overlaps) using direct geometry.
    The hexagons will be used to represent phase elements.
    
    Parameters:
      plate_size: Physical size of the phase plate.
      element_size: Horizontal corner-to-corner width of each hexagon.
    
    Returns:
      A NumPy array of shape (N_hex, 6, 2) where each hexagon is defined by its 6 vertices (x, y).
    """
    # Compute side length (distance from the hexagon center to a vertex)
    s = element_size / 2.0
    
    # Define the vertex angles for a flat-top hexagon (in degrees)
    angles_deg = [30, 90, 150, 210, 270, 330]
    angles_rad = np.radians(angles_deg)  # Convert angles to radians
    
    # Compute the spacing between hexagon centers
    dx = 1.745 * s          # Horizontal spacing
    dy = np.float16(np.sqrt(3 - 0.71) * s)  # Vertical spacing using an empirical adjustment factor
    offset_x = dx / 2.0       # Horizontal offset for every other (odd) row to create the staggered grid

    # Define the bounding box centered at (0,0)
    x_min, x_max = -plate_size / 2, plate_size / 2
    y_min, y_max = -plate_size / 2, plate_size / 2

    # Determine the number of rows and columns needed to cover the plate
    nrows = int(np.ceil((y_max - y_min) / dy)) + 2
    ncols = int(np.ceil((x_max - x_min) / dx)) + 2

    hex_list = []  # List to hold all hexagon vertex arrays
    for row in range(nrows):
        center_y = y_min + row * dy  # Y-coordinate of the hexagon center for this row
        for col in range(ncols):
            center_x = x_min + col * dx  # X-coordinate for the current column
            # For odd rows, shift the x-coordinate by the offset
            if row % 2 == 1:
                center_x += offset_x

            # Skip hexagons that fall outside the defined bounding box (with a margin)
            if center_x < x_min - dx or center_x > x_max + dx:
                continue
            if center_y < y_min - dy or center_y > y_max + dy:
                continue

            # Compute the six vertices for the hexagon
            hex_verts = []
            for theta in angles_rad:
                vx = center_x + s * np.cos(theta)
                vy = center_y + s * np.sin(theta)
                hex_verts.append((vx, vy))
            hex_list.append(hex_verts)

    # Convert the list of vertices into a NumPy array for further use
    return np.array(hex_list)  # Shape: (N_hex, 6, 2)

def plot_phase_map_with_hex_overlay(phase_map, plate_size, hex_array):
    """
    Plot the phase map with an overlay of a hexagonal grid and a circular boundary
    that represents the physical phase plate.
    
    Parameters:
      phase_map: 2D array of phase values (in radians).
      plate_size: Physical diameter of the phase plate (m).
      hex_array: Array of hexagon vertices with shape (N_hex, 6, 2).
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    # Set plot extent to match the physical dimensions of the plate
    extent = (-plate_size/2, plate_size/2, -plate_size/2, plate_size/2)
    # Display the phase map using an HSV colormap for phase visualization
    im = ax.imshow(phase_map, extent=extent, origin='upper', cmap='hsv')
    fig.colorbar(im, ax=ax, label='Phase (radians)')
    
    # Overlay each hexagon on the phase map
    for hex_verts in hex_array:
        patch = Polygon(hex_verts, closed=True, edgecolor='k', facecolor='none', lw=1)
        ax.add_patch(patch)
    
    # Draw a red circular boundary to denote the plate edge
    circle = Circle((0, 0), plate_size/2, edgecolor='red', facecolor='none', lw=2)
    ax.add_patch(circle)
    
    ax.set_xlim(-plate_size/2, plate_size/2)
    ax.set_ylim(-plate_size/2, plate_size/2)
    ax.set_aspect('equal')
    ax.set_title("Phase Map with Hexagonal Grid Overlay")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    plt.show()

def apply_circular_mask(image):
    """
    Apply a circular mask to an image array, preserving values only within the circle.
    
    Parameters:
      image: 2D numpy array representing an image.
      
    Returns:
      mask: 2D numpy array with the same shape as 'image' where values outside the circle are zero.
    """
    # Initialize an array of zeros with the same shape as the image
    mask = np.zeros(image.shape)
    # Calculate the center of the image
    center = (image.shape[0] // 2, image.shape[1] // 2)
    # Determine the radius as half the minimum dimension of the image
    radius = min(image.shape) // 2
    # Generate a grid of indices for the image
    Y, X = np.ogrid[:image.shape[0], :image.shape[1]]
    # Create a boolean array that is True inside the circle and False outside
    mask_area = (X - center[1])**2 + (Y - center[0])**2 <= radius**2
    # Copy the original image values to the mask only where the condition is True
    mask[mask_area] = image[mask_area]
    return mask

def gs_2d(n: int, amp: float, beam_fwhm: float, mod_amp: float, mod_freq: float,
          plate_size=0.2, max_iter=100, discrete=True, plot=False, pzp=False, nsteps=2, 
          wavelength=532e-9, focal_length=1.0, refractive_index=1.5, discretize=False, 
          num_steps=10, desired_focal_spot=500e-6, focal_spot_size=500e-6, 
          input_beam_file: str = None, file_type: str = "npy", input_theta_file = None, file_type_theta = None) -> np.ndarray:
    """
    Gerchberg-Saxton (GS) algorithm for designing a phase plate.
    Iteratively retrieves phase information in the Fourier domain to converge on a phase profile.
    
    Parameters:
      n: Grid resolution for phase calculation.
      amp: Input beam amplitude (arbitrary units).
      std: Standard deviation for the Gaussian beam profile (m).
      mod_amp: Modulation amplitude.
      mod_freq: Modulation frequency.
      plate_size: Physical diameter of the phase plate (m).
      max_iter: Maximum number of iterations for the algorithm.
      discrete: Boolean flag to indicate if discretization should be applied.
      plot: Boolean flag to enable plotting during iterations.
      pzp, nsteps: Additional parameters (not actively used here).
      wavelength: Laser wavelength (m).
      focal_length: Focal length of the optical system (m).
      refractive_index: Refractive index of the phase plate material.
      discretize: Boolean flag for discretizing the phase values.
      num_steps: Number of discrete phase steps (if discretization is applied).
      desired_focal_spot: Desired focal spot size (m).
      focal_spot_size: Focal spot size used in simulation (m).
      input_beam_file: Optional file path for the input beam intensity.
      file_type: Type of file for input beam ('npy' or 'image').
      input_theta_file: Optional file path for an initial phase (theta) distribution.
      file_type_theta: File type for the theta file.
    
    Returns:
      theta_in: The final computed phase profile (in radians) as a 2D array.
    """

    std = beam_fwhm / (2 * np.sqrt(2 * np.log(2)))

    # Set the grid resolution (number of variable phase points)
    num_elements = n
     
    # Create a spatial grid spanning from -plate_size/2 to plate_size/2 in both x and y
    x = np.linspace(-plate_size / 2, plate_size / 2, num_elements)
    # Create an array to hold the (x,y) coordinates at each grid point
    xy = np.zeros((len(x), len(x), 2))
    for i, row in enumerate(xy):
        for j, _ in enumerate(row):
            xy[i, j] = [x[i], x[j]]  # Assign the x and y coordinates

    # Initialize lists to store iteration data and convergence metrics
    i_arr = []
    convergence = []

    # If an input beam file is provided, load it using the appropriate helper function;
    # otherwise, generate a default Gaussian beam profile.
    if input_beam_file:
        print(f"Loading input beam intensity from {input_beam_file} as a {file_type} file...")
        if file_type == "numpy":
            input_beam = load_input_beam(input_beam_file, (num_elements, num_elements))
        elif file_type == "image":
            input_beam = load_input_beam_png(input_beam_file, (num_elements, num_elements))
        else:
            raise ValueError("Unsupported file_type. Use 'numpy' or 'image'.")
    else:
        input_beam = np.exp(- (np.linalg.norm(xy, axis=2) / std)**2)

    # Define the input beam as a Gaussian profile
    input_beam = np.exp(- (np.linalg.norm(xy, axis=2) / std)**2)
    # Define the ideal beam as a super-Gaussian (with sharper edges) for target comparison
    ideal_beam = np.exp(- (np.linalg.norm(xy, axis=2) / std)**4)
    
    # Optionally plot the input and ideal beam profiles
    if plot:
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        im1 = ax1.imshow(input_beam, extent=[-plate_size/2, plate_size/2, -plate_size/2, plate_size/2], cmap='gray')
        ax1.set_title('Scaled Input Gaussian Beam')
        ax1.set_xlabel('x (m)')
        ax1.set_ylabel('y (m)')
        cbar = fig1.colorbar(im1, ax=ax1)
        cbar.ax.set_title('$Wm^{-2}$')
        
        im2 = ax2.imshow(ideal_beam, extent=[-plate_size/2, plate_size/2, -plate_size/2, plate_size/2], cmap='gray')
        ax2.set_title('Scaled Ideal Super Gaussian Beam')
        ax2.set_xlabel('x (m)')
        ax2.set_ylabel('y (m)')
        cbar = fig1.colorbar(im2, ax=ax2)
        cbar.ax.set_title('$Wm^{-2}$')
        plt.show()

    # If an initial theta file is provided, load it; otherwise, generate a random phase distribution.
    if input_theta_file:
        print(f"Loading input beam intensity from {input_theta_file} as a {file_type_theta} file...")
        if file_type == "numpy":
            input_beam = load_input_beam(input_theta_file, (num_elements, num_elements))
        elif file_type == "image":
            input_beam = load_input_beam_png(input_theta_file, (num_elements, num_elements))
        else:
            raise ValueError("Unsupported file_type. Use 'numpy' or 'image'.")
    else:
        # Create a random phase distribution using multiples of pi/2 over the grid
        theta_in = (np.pi / 2) * np.random.randint(-2, 3, size=np.shape(xy)[:-1])

    # Get the amplitude (electric field magnitude) of the input beam
    original_beam_electric = np.abs(input_beam)

    # Main iterative loop for the Gerchberg-Saxton algorithm
    for i in range(max_iter):
        # Print progress every 5% of the iterations
        if i % (max_iter // 20) == 0:
            print(f"GS algorithm: {int((i / max_iter) * 100)} % complete")

        # Form the complex electric field using the current phase (theta_in) and the amplitude
        input_beam_electric = np.square(original_beam_electric) * np.exp(1j * theta_in)
        # Compute the Fourier transform to simulate propagation into the focal plane
        beam_ft = np.fft.fft2(input_beam_electric)
        # Normalize the Fourier transform and match its maximum amplitude to the ideal beam
        beam_ft = beam_ft / np.max(beam_ft) * np.max(ideal_beam)
        # Extract the phase information from the Fourier domain
        theta_out = np.angle(beam_ft)

        # Combine the ideal amplitude (super-Gaussian) with the new phase to form a new beam in the focal plane
        new_beam_ft = np.square(ideal_beam) * np.exp(1j * theta_out)
        # Inverse Fourier transform to propagate back to the input plane
        new_beam_electric = np.fft.ifft2(new_beam_ft)
        # Update theta_in with the phase from the inverse-transformed field
        theta_in = np.angle(new_beam_electric)

        # Record iteration number and convergence metric (difference between current Fourier amplitude and ideal)
        i_arr.append(i)
        convergence.append(np.sum(np.abs(np.abs(beam_ft) - ideal_beam)) / np.sum(ideal_beam))

    # Adjust phase values to be within the range [0, 2π)
    theta_in = np.where(theta_in < 0, theta_in + 2 * np.pi, theta_in)
    theta_in = np.where(theta_in >= 2 * np.pi, theta_in - 2 * np.pi, theta_in)

    # If discretization is requested, round the phase values to discrete steps
    if discretize:
        step_size = 2 * np.pi / num_steps
        theta_in = np.round(theta_in / step_size) * step_size
        print(f"Phase plate discretized into {num_steps} steps.")

    print(f"Continuous convergence accuracy: {100 - convergence[-1] * 100:.2f} %")

    # Compute the material thickness required to achieve the phase shift (using the refractive index)
    thickness = theta_in * wavelength / (2 * np.pi * (refractive_index - 1))

    # Simulate the beam after it passes through the phase plate by applying the computed phase shift
    phase_plate_electric = np.abs(input_beam) * np.exp(1j * theta_in)
    # Compute the Fourier transform of the modified beam to obtain the focal spot intensity
    focal_spot = np.fft.fftshift(np.fft.fft2(phase_plate_electric))
    focal_spot = np.abs(focal_spot) ** 2
    
    # Plot various intermediate and final results if requested
    if plot:
        fig2, axes = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)

        im0 = axes[0, 0].imshow(apply_circular_mask(np.abs(original_beam_electric)), cmap='gray', 
                          extent=[-plate_size/2, plate_size/2, -plate_size/2, plate_size/2])
        axes[0, 0].set_title('Input Beam (Gaussian)')
        axes[0, 0].set_xlabel('x (m)')
        axes[0, 0].set_ylabel('y (m)')
        cbar = fig2.colorbar(im0, ax=axes[0, 0], label='$Wm^{-2}$')

        im1 = axes[0, 1].imshow(apply_circular_mask(np.abs(ideal_beam)), cmap='gray', 
                          extent=[-plate_size/2, plate_size/2, -plate_size/2, plate_size/2])
        axes[0, 1].set_title('Ideal Beam (Super Gaussian)')
        axes[0, 1].set_xlabel('x (m)')
        axes[0, 1].set_ylabel('y (m)')
        cbar = fig2.colorbar(im1, ax=axes[0, 1], label='$Wm^{-2}$')

        im2 = axes[0, 2].imshow(apply_circular_mask(np.abs(new_beam_electric)), cmap='gray', 
                          extent=[-plate_size/2, plate_size/2, -plate_size/2, plate_size/2])
        axes[0, 2].set_title('Output Beam')
        axes[0, 2].set_xlabel('x (m)')
        axes[0, 2].set_ylabel('y (m)')
        cbar = fig2.colorbar(im2, ax=axes[0, 2], label='$Wm^{-2}$')

        im3 = axes[1, 0].imshow(apply_circular_mask(np.log(focal_spot)), cmap='inferno', 
                          extent=[-focal_spot_size/2 *1e6, focal_spot_size/2 *1e6, -focal_spot_size/2 *1e6, focal_spot_size/2 *1e6])
        axes[1, 0].set_title('Focal Spot after Phase Plate')
        axes[1, 0].set_xlabel('x ($\mu$m)')
        axes[1, 0].set_ylabel('y ($\mu$m)')
        cbar = fig2.colorbar(im3, ax=axes[1, 0], label='$Wm^{-2}$')

        im4 = axes[1, 1].imshow(apply_circular_mask(theta_in), cmap='hsv', 
                                 extent=[-plate_size/2, plate_size/2, -plate_size/2, plate_size/2])
        axes[1, 1].set_title('Generated Phase Plate')
        axes[1, 1].set_xlabel('x (m)')
        axes[1, 1].set_ylabel('y (m)')
        cbar = fig2.colorbar(im4, ax=axes[1, 1], label='Phase (radians)')

        im5 = axes[1, 2].imshow(apply_circular_mask(thickness) *1e6, cmap='gray', 
                                 extent=[-plate_size/2, plate_size/2, -plate_size/2, plate_size/2])
        axes[1, 2].set_title('Material Thickness (m)')
        axes[1, 2].set_xlabel('x (m)')
        axes[1, 2].set_ylabel('y (m)')
        cbar = fig2.colorbar(im5, ax=axes[1, 2], label='$\mu$m' )
       
        plt.show()

    # Return the computed phase map (theta)
    return theta_in

def plot_phase_map_with_hex_overlay(phase_map, plate_size, hex_array):
    """
    Plot the phase map (phase plate) with an overlay of the hexagonal grid.
    Also draws a circular boundary corresponding to the plate.
    
    Parameters:
      phase_map: 2D numpy array of phase values (radians).
      plate_size: Physical plate size (m).
      hex_array: Array of hexagon vertices (shape: (N_hex, 6, 2)).
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    extent = (-plate_size/2, plate_size/2, -plate_size/2, plate_size/2)
    im = ax.imshow(phase_map, extent=extent, origin='upper', cmap='inferno')
    fig.colorbar(im, ax=ax, label='Phase (radians)')
    
    # Overlay each hexagon from the hex_array onto the phase map
    for hex_verts in hex_array:
        patch = Polygon(hex_verts, closed=True, edgecolor='k', facecolor='none', lw=1)
        ax.add_patch(patch)
    
    # Draw a circular boundary (red) to indicate the plate edge
    circle = Circle((0, 0), plate_size/2, edgecolor='red', facecolor='none', lw=2)
    ax.add_patch(circle)
    
    ax.set_xlim(-plate_size/2, plate_size/2)
    ax.set_ylim(-plate_size/2, plate_size/2)
    ax.set_aspect('equal')
    ax.set_title("Phase Map with Phase Element Overlay")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    plt.show()

def smooth_theta(theta, plate_size, order=4, smoothing=0):
    """
    Smooth the phase plate 'theta' using a bivariate spline interpolation.
    
    Parameters:
      theta: 2D numpy array of phase values (radians).
      plate_size: Physical size of the phase plate (m).
      order: Order of the spline (default 4th order).
      smoothing: Smoothing factor (set to 0 to interpolate exactly through points).
    
    Returns:
      theta_smooth: 2D numpy array of smoothed phase values evaluated on a finer grid.
    """
    nrows, ncols = theta.shape
    # Create coordinate arrays that span from -plate_size/2 to plate_size/2
    x = np.linspace(-plate_size / 2, plate_size / 2, ncols)
    y = np.linspace(-plate_size / 2, plate_size / 2, nrows)
    
    # Create the bivariate spline with specified order and smoothing.
    # Note: The first coordinate (y) comes first since the input array is (rows, cols)
    spline = RectBivariateSpline(y, x, theta, kx=order, ky=order, s=smoothing)
    
    # Retrieve and print the coefficients of the spline for inspection
    coeffs = spline.get_coeffs()
    print("Spline coefficients shape:", coeffs.shape)
    print("Spline coefficients:", coeffs)

    # Evaluate the spline on a finer grid to produce a smooth output
    new_grid_x = np.linspace(-plate_size / 2, plate_size / 2, ncols*10)
    new_grid_y = np.linspace(-plate_size / 2, plate_size / 2, nrows*10)
    theta_smooth = spline(new_grid_y, new_grid_x)
    return theta_smooth


def compute_focal_spot_with_smoothed_phase(theta_smoothed, plate_size, desired_focal_spot, 
                                           input_beam_filepath=None, file_type="npy", beam_fwhm=None):
    """
    Compute and visualize the focal spot after propagation through a smoothed phase plate.

    This function performs the following steps:
      1. Constructs a fine coordinate grid based on the resolution of theta_smoothed.
      2. Loads an input beam from file (if provided) and upscales it to match the fine grid. 
         If no beam file is provided, a Gaussian beam is generated using a FWHM.
         The FWHM is converted to the standard deviation sigma via:
             sigma = beam_fwhm / (2 * sqrt(2 * ln(2)))
         By default, beam_fwhm is set to the plate_size, ensuring the beam is comparable to 
         the physical extent of the phase plate.
      3. Applies the smoothed phase (theta_smoothed) to the beam.
      4. Propagates the field via a Fourier transform to compute the focal spot.
      5. Plots three panels:
           - The high-resolution input beam intensity.
           - The smoothed phase plate.
           - The focal spot intensity (with an extent based on desired_focal_spot).

    Parameters:
      theta_smoothed (np.ndarray): 2D array of the smoothed phase plate (radians).
      plate_size (float): Diameter of the phase plate (m).
      desired_focal_spot (float): Size of the desired focal spot (m) for plotting.
      input_beam_filepath (str, optional): Path to the stored input beam file.
      file_type (str): Type of file to load ("npy" or "image"). Default is "npy".
      beam_fwhm (float, optional): Full-width at half-maximum of the Gaussian beam (m). 
                                   If None, defaults to plate_size.

    Returns:
      np.ndarray: The computed focal intensity distribution.
    """

    # Set default beam FWHM if not provided.
    if beam_fwhm is None:
        beam_fwhm = plate_size  # Default: FWHM comparable to the plate size

    # Retrieve fine grid dimensions from the smoothed phase plate.
    nrows, ncols = theta_smoothed.shape

    # Construct a fine coordinate grid that spans the plate.
    x_fine = np.linspace(-plate_size / 2, plate_size / 2, ncols)
    y_fine = np.linspace(-plate_size / 2, plate_size / 2, nrows)
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine)

    # Load and upscale the input beam if a file is provided.
    if input_beam_filepath is not None:
        if file_type.lower() in ["npy", "numpy"]:
            native_beam = np.load(input_beam_filepath)
        elif file_type.lower() == "image":
            native_beam = cv2.imread(input_beam_filepath, cv2.IMREAD_GRAYSCALE)
            if native_beam is None:
                raise ValueError(f"Could not read file: {input_beam_filepath}")
        else:
            raise ValueError("Unsupported file_type. Use 'npy' or 'image'.")
        
        if native_beam.ndim > 2:
            native_beam = np.squeeze(native_beam)
        native_beam = native_beam.astype(np.float64)
        
        beam_high_res = cv2.resize(native_beam, (ncols, nrows), interpolation=cv2.INTER_LINEAR)
        beam_high_res = beam_high_res / np.max(beam_high_res)
        print("Input beam loaded and upscaled from shape {} to {}."
              .format(native_beam.shape, beam_high_res.shape))
    else:
        # Generate a Gaussian beam based on the provided FWHM.
        # Convert FWHM to sigma: sigma = FWHM / (2 * sqrt(2 * ln(2))).
        sigma = beam_fwhm / (2 * np.sqrt(2 * np.log(2)))
        beam_high_res = np.exp(- (X_fine**2 + Y_fine**2) / (2 * sigma**2))
        print("No input beam file provided. Generated a Gaussian beam with FWHM = {:.3e} m on the fine grid."
              .format(beam_fwhm))
    
    # Form the complex field by applying the smoothed phase.
    phase_field_smoothed = beam_high_res * np.exp(1j * theta_smoothed)
    
    # Propagate to the focal plane via Fourier transform.
    focal_field = np.fft.fftshift(np.fft.fft2(phase_field_smoothed))
    focal_intensity_smoothed = np.abs(focal_field)**2
    
    # Plot the three components.
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot the high-resolution input beam.
    im0 = axs[0].imshow(apply_circular_mask(beam_high_res), extent=[-plate_size/2, plate_size/2,
                                                 -plate_size/2, plate_size/2],
                          cmap='gray')
    axs[0].set_title("High-Resolution Input Beam (Gaussian)")
    axs[0].set_xlabel("x (m)")
    axs[0].set_ylabel("y (m)")
    fig.colorbar(im0, ax=axs[0], shrink=0.8, label="Normalized Intensity")
    
    # Plot the smoothed phase plate.
    im1 = axs[1].imshow(apply_circular_mask(theta_smoothed), extent=[-plate_size/2, plate_size/2,
                                                  -plate_size/2, plate_size/2],
                          cmap='hsv')
    axs[1].set_title("Smoothed Phase Plate")
    axs[1].set_xlabel("x (m)")
    axs[1].set_ylabel("y (m)")
    fig.colorbar(im1, ax=axs[1], shrink=0.8, label="Phase (radians)")
    
    # Plot the focal spot intensity.
    im2 = axs[2].imshow(np.log(apply_circular_mask(focal_intensity_smoothed)),
                          extent=[-desired_focal_spot/2, desired_focal_spot/2,
                                  -desired_focal_spot/2, desired_focal_spot/2],
                          cmap='inferno')
    axs[2].set_title("Focal Spot (Log Intensity)")
    axs[2].set_xlabel("x (m)")
    axs[2].set_ylabel("y (m)")
    fig.colorbar(im2, ax=axs[2], shrink=0.8, label="Log Intensity")
    
    plt.tight_layout()
    plt.show()
    
    np.save("focal_intensity_smoothed.npy", focal_intensity_smoothed)
    return focal_intensity_smoothed

# ---------------------------------------------------
# Main execution block: runs if the script is executed directly
if __name__ == "__main__":
    # Simulation parameters
    n = 1000                  # Grid resolution for phase calculation
    amp = 8.0                 # Input beam amplitude (arbitrary units)
    std = 0.03                # Standard deviation for the Gaussian beam profile (m)
    beam_fwhm = 15e-2         # Full-width at half-maximum of the Gaussian beam (m)       
    mod_amp = 0.5             # Modulation amplitude
    mod_freq = 10.0           # Modulation frequency
    plate_size = 0.2          # Physical diameter of the phase plate (m)
    max_iter = 100            # Number of iterations for the Gerchberg-Saxton algorithm
    wavelength = 532e-9       # Laser wavelength (m)
    focal_length = 1.0        # Focal length of the optical system (m)
    refractive_index = 1.5    # Refractive index of the phase plate material
    desired_focal_spot = 500e-7  # Desired focal spot size (m)

    # Compute phase element parameters based on the simulation and physical parameters
    element_size, num_elements, element_area, focal_spot_size = compute_phase_elements(
        plate_size, wavelength, focal_length, desired_focal_spot
    )
    
    # Build the hexagonal grid geometry (used later for overlay visualization)
    hex_array = build_flat_top_hex_grid(plate_size, element_size)
    print("Hex array shape:", hex_array.shape)

    # Create and display a hex grid image using OpenCV 
    img_size = (n, n)
    result_img = cv2.cvtColor(cv2.bitwise_and(
        np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 255,
        np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 255,
        mask=cv2.circle(np.zeros((img_size[1], img_size[0]), dtype=np.uint8),
                        (img_size[0]//2, img_size[1]//2), img_size[0]//2, 255, -1)
    ), cv2.COLOR_BGR2RGB)

    # Uncomment below lines to display the hex grid image:
    # plt.figure(figsize=(6,6))
    # plt.imshow(result_img); plt.title("Hex Grid (OpenCV)"); plt.axis("off"); plt.show()

    # Define file paths for input beam and initial phase (if available); here, we use defaults (None)
    input_beam_filepath = None
    input_theta_filepath = None

    # Compute the phase plate using the Gerchberg-Saxton algorithm
    theta_in = gs_2d(n, amp, beam_fwhm, mod_amp, mod_freq,
                     plate_size=plate_size, max_iter=max_iter, plot=True, wavelength=wavelength,
                     focal_length=focal_length, refractive_index=refractive_index,
                     discretize=True, num_steps=1e10, desired_focal_spot=desired_focal_spot, 
                     focal_spot_size=focal_spot_size, input_beam_file=input_beam_filepath, file_type=None,
                     input_theta_file=input_theta_filepath, file_type_theta=None)
    
    # Overlay the hexagonal grid on the computed phase plate and display the result
    plot_phase_map_with_hex_overlay(apply_circular_mask(theta_in), plate_size, hex_array)

    # Smooth the phase plate using a bivariate spline to obtain a continuous phase function
    theta_in_smoothed = smooth_theta(theta_in, plate_size, order=3, smoothing=0)

    #recalculate the focal spot using the smoothed phase
    focal_intensity_smoothed = compute_focal_spot_with_smoothed_phase(
        theta_in_smoothed, plate_size, desired_focal_spot, input_beam_filepath=input_beam_filepath, file_type=None, beam_fwhm = beam_fwhm
    )
   
    # Record the end time and calculate the total execution time
    end = time.time()

    # Display the smoothed phase plate as an image
    plt.imshow(apply_circular_mask(theta_in_smoothed))
    plt.show()

    plt.imshow(focal_intensity_smoothed, cmap='inferno')
    plt.title("Focal Spot Intensity with Smoothed Phase Plate")
    plt.show()

    print("Total time =", end-start, "seconds")
