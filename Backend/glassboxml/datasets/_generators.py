import numpy as np

def make_regression(n_samples=100, n_features=1, noise=0.1, random_state=None):
    """
    Generates a random linear dataset for regression testing.
    Returns: X (features), y (target), true_weights, true_bias
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    X = 2 * np.random.rand(n_samples, n_features)
    
    # Generate random true underlying weights and bias
    true_weights = np.random.uniform(-5, 5, n_features)
    true_bias = np.random.uniform(-5, 5)
    
    # y = Xw + b + noise
    y = np.dot(X, true_weights) + true_bias
    y += noise * np.random.randn(n_samples)
    
    return X, y, true_weights, true_bias


import numpy as np


def make_classification(n_samples=100, n_features=2, n_classes=2, random_state=None):
    """
    Generate a simple random classification dataset.
    
    Parameters
    ----------
    n_samples : int, default=100
        Number of samples to generate.
        
    n_features : int, default=2
        Number of features.
        
    n_classes : int, default=2
        Number of classes.
        
    random_state : int or None, default=None
        Random seed for reproducibility.
        
    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
        
    y : ndarray of shape (n_samples,)
        Target labels (0 to n_classes-1).
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate samples for each class
    X = []
    y = []
    
    samples_per_class = n_samples // n_classes
    
    for class_idx in range(n_classes):
        # Create cluster center for this class
        center = np.random.randn(n_features) * 3 + class_idx * 5
        
        # Handle remainder samples in the last class
        if class_idx == n_classes - 1:
            n_class_samples = n_samples - class_idx * samples_per_class
        else:
            n_class_samples = samples_per_class
        
        # Generate samples around the center
        class_samples = np.random.randn(n_class_samples, n_features) + center
        
        X.append(class_samples)
        y.append(np.full(n_class_samples, class_idx))
    
    # Combine all classes
    X = np.vstack(X)
    y = np.concatenate(y)
    
    # Shuffle the dataset
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    return X, y


def make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=1.0, random_state=None):
    """
    Generates distinct clusters of points for classification or clustering testing.
    Returns: X (features), y (cluster labels)
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    X = []
    y = []
    
    samples_per_center = n_samples // centers
    
    for i in range(centers):
        # Pick a random center point in space
        center_coords = np.random.uniform(-10, 10, n_features)
        
        # Generate a cloud of points around that center
        cluster_points = center_coords + cluster_std * np.random.randn(samples_per_center, n_features)
        
        X.append(cluster_points)
        y.append(np.full(samples_per_center, i))
        
    # Stack everything into neat arrays
    X = np.vstack(X)
    y = np.concatenate(y)
    
    # Shuffle the dataset so classes aren't strictly ordered
    shuffle_indices = np.random.permutation(len(X))
    return X[shuffle_indices], y[shuffle_indices]


def make_stretched_blobs(n_samples_per_class=150, random_state=None):
    """
    Generates 3 classes of 3D data for dimensionality reduction testing.
    The classes are perfectly separated along the Y-axis, but massively stretched 
    along the X-axis. This acts as a trap to test if an algorithm prioritizes 
    pure variance (PCA) or class separation (LDA).
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    # Class 0 (Top)
    x0 = np.random.normal(0, 15, n_samples_per_class)  # Massive X variance
    y0 = np.random.normal(5, 1, n_samples_per_class)   # Y = 5
    z0 = np.random.normal(0, 1, n_samples_per_class)
    
    # Class 1 (Middle)
    x1 = np.random.normal(0, 15, n_samples_per_class)
    y1 = np.random.normal(0, 1, n_samples_per_class)   # Y = 0
    z1 = np.random.normal(0, 1, n_samples_per_class)
    
    # Class 2 (Bottom)
    x2 = np.random.normal(0, 15, n_samples_per_class)
    y2 = np.random.normal(-5, 1, n_samples_per_class)  # Y = -5
    z2 = np.random.normal(0, 1, n_samples_per_class)
    
    X = np.vstack([np.column_stack([x0, y0, z0]), 
                   np.column_stack([x1, y1, z1]), 
                   np.column_stack([x2, y2, z2])])
    
    y = np.concatenate([np.zeros(n_samples_per_class), 
                        np.ones(n_samples_per_class), 
                        np.full(n_samples_per_class, 2)])
    
    return X, y

def make_donut(n_samples=200, noise=0.1, random_state=None):
    """
    Generates a 2D 'donut' dataset (a circle inside a ring).
    This data is impossible to separate with a straight linear line.
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    # Inner circle (Class 0)
    n_inner = n_samples // 2
    theta_inner = np.random.uniform(0, 2 * np.pi, n_inner)
    r_inner = 1.0 + np.random.normal(0, noise, n_inner)
    
    # Outer circle (Class 1)
    n_outer = n_samples - n_inner
    theta_outer = np.random.uniform(0, 2 * np.pi, n_outer)
    r_outer = 3.0 + np.random.normal(0, noise, n_outer)
    
    # Combine
    X = np.vstack([
        np.column_stack([r_inner * np.cos(theta_inner), r_inner * np.sin(theta_inner)]),
        np.column_stack([r_outer * np.cos(theta_outer), r_outer * np.sin(theta_outer)])
    ])
    y = np.concatenate([np.zeros(n_inner), np.ones(n_outer)])
    
    return X, y


def make_moons(n_samples=200, noise=0.1, random_state=None):
    """
    Generates a 2D dataset of two interlocking half-circles (moons).
    A classic non-linear dataset to test complex boundaries.
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out
    
    # Upper moon (Class 0)
    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
    
    # Lower moon (Class 1), shifted over and down
    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
    inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - 0.5
    
    X = np.vstack([np.append(outer_circ_x, inner_circ_x),
                   np.append(outer_circ_y, inner_circ_y)]).T
    
    y = np.hstack([np.zeros(n_samples_out, dtype=int),
                   np.ones(n_samples_in, dtype=int)])
                   
    # Add random Gaussian noise
    if noise is not None:
        X += np.random.normal(scale=noise, size=X.shape)
        
    return X, y



def make_circles(n_samples=100, factor=0.8, noise=None, random_state=None):
    """
    Generates a large circle containing a smaller circle in 2D.
    A pure-Numpy replacement for sklearn.datasets.make_circles.
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    # 1. Split the data points evenly between the two circles
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out
    
    # 2. Generate evenly spaced angles from 0 to 2*pi
    linspace_out = np.linspace(0, 2 * np.pi, n_samples_out, endpoint=False)
    linspace_in = np.linspace(0, 2 * np.pi, n_samples_in, endpoint=False)
    
    # 3. Calculate coordinates for the Outer Circle (radius = 1.0)
    outer_circ_x = np.cos(linspace_out)
    outer_circ_y = np.sin(linspace_out)
    
    # 4. Calculate coordinates for the Inner Circle (radius = factor)
    inner_circ_x = factor * np.cos(linspace_in)
    inner_circ_y = factor * np.sin(linspace_in)
    
    # 5. Stack the X and Y coordinates together into a single dataset
    X = np.vstack([
        np.append(outer_circ_x, inner_circ_x),
        np.append(outer_circ_y, inner_circ_y)
    ]).T
    
    # 6. Create the labels (0 for outer ring, 1 for inner core)
    y = np.hstack([
        np.zeros(n_samples_out, dtype=int),
        np.ones(n_samples_in, dtype=int)
    ])
    
    # 7. Sprinkle in the Gaussian noise to make it messy
    if noise is not None:
        X += np.random.normal(scale=noise, size=X.shape)
        
    # 8. Shuffle the dataset so the model doesn't just read it in order
    indices = np.random.permutation(n_samples)
    
    return X[indices], y[indices]