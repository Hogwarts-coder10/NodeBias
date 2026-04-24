import numpy as np

def train_test_split(*arrays, test_size=0.2, random_state=None, shuffle=True):
    """
    Splits an arbitrary number of arrays or matrices into random train and test subsets.
    This prevents the model from 'memorizing'.
    """
    if len(arrays) == 0:
        raise ValueError("At least one array required as input")

    # Grab the number of rows from the first array passed in
    n_samples = len(arrays[0])

    # Quick safety check: ensure all passed arrays have the same number of rows
    for arr in arrays:
        if len(arr) != n_samples:
            raise ValueError("All arrays must have the same number of samples!")

    if random_state is not None:
        np.random.seed(random_state)

    indices = np.arange(n_samples)

    # Always shuffle before splitting so we get a healthy mix of classes!
    if shuffle:
        np.random.shuffle(indices)

    # Calculate the exact row index where we need to slice the data
    split_idx = int(n_samples * (1 - test_size))

    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    # Slice every matrix that was passed in!
    result = []
    for arr in arrays:
        # Convert to numpy array just in case a Pandas Series is passed
        arr_np = np.array(arr)

        train_part = arr_np[train_indices]
        test_part = arr_np[test_indices]

        # Append train then test (e.g., X_train, X_test, y_train, y_test, gender_train, gender_test)
        result.extend([train_part, test_part])

    return result
