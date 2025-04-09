import numpy as np

def euclidean_distance_matrix(X: np.array, Y: np.array) -> np.array:
    """
    Calculate the Euclidean distance between all rows in matrices X and Y.

   :param X: numpy array, shape (n_samples1, n_features)
   :param Y: numpy array, shape (n_samples2, n_features)
   :return: numpy array, shape (n_samples1, n_samples2)
               The Euclidean distance array where D[i, j] is the Euclidean distance
               between the ith row of X and the jth row of Y.
    """
    # Reshape input matrices to ensure proper broadcasting
    X = X.reshape(X.shape[0], 1, -1)
    Y = Y.reshape(1, Y.shape[0], -1)

    # Calculate Euclidean distance
    D = np.sqrt(np.sum((X - Y) ** 2, axis=-1))

    return D
