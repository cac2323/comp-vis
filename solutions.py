import numpy as np
import utils
from typing import Tuple

import matplotlib.pyplot as plt

ransac_n = 500
ransac_eps = 2
window_size = 21

def sign_academic_honesty_policy():
    print("I, %s (%s), certify that I have read and agree to the Code of Academic Integry." %
        ("Canyon Clark", "cac2323"))


def estimate_P(uv: np.ndarray, X_world: np.ndarray) -> np.ndarray:
    """
    Estimate the projection matrix P from the given point correspondences.

    Args:
        uv (np.ndarray): image coordinates in pixels (Nx2)
        X_world (np.ndarray): world coordinates in emters (Nx3)

    Returns:
        np.ndarray: camera projection matrix (3x4)
    """

    assert uv.shape[0] == X_world.shape[0]
    assert uv.shape[1] == 2 and X_world.shape[1] == 3

    # print("uv: ", uv.shape)
    # print("world: ", X_world.shape)

    u = uv[:, 0]
    v = uv[:, 1]

    xw = X_world[:,0]
    yw = X_world[:,1]
    zw = X_world[:,2]

    A = []

    for i in range(uv.shape[0]):

        A.append([xw[i], yw[i], zw[i], 1, 0, 0, 0, 0, -u[i]*xw[i], -u[i]*yw[i], -u[i]*zw[i], -u[i]])
        A.append([0, 0, 0, 0, xw[i], yw[i], zw[i], 1, -v[i]*xw[i], -v[i]*yw[i], -v[i]*zw[i], -v[i]])

    A = np.array(A)

    U, S, V = np.linalg.svd(A)
    P = V[-1, :].reshape(3, 4)

    return P


def reprojection_error(uv: np.ndarray, 
                       X_world: np.ndarray, 
                       P: np.ndarray) -> float:
    """
    Compute the root-mean-squared (RMS) reprojection error over all the points.

    Args:
        uv (np.ndarray): image points (Nx2)
        X_world (np.ndarray): world points (Nx3)
        P (np.ndarray): camera projection matrix (3x4)

    Returns:
        float: RMS reprojection error
    """

    X_homogeneous = np.hstack((X_world, np.ones((X_world.shape[0], 1))))

    new_homogeneous = np.dot(P, X_homogeneous.T).T

    projected_points = new_homogeneous[:, :2] / new_homogeneous[:, 2, np.newaxis]

    squared_errors = np.sum((uv - projected_points) ** 2, axis=1)

    # MSE
    mean_squared_error = np.mean(squared_errors)

    #  RMSE
    error = np.sqrt(mean_squared_error)

    return error

def estimate_F(uv1: np.ndarray, uv2: np.ndarray) -> np.ndarray:
    """
    Estimate the fundamental matrix mapping the points in uv1 and
    uv2. The fundamental matrix should map points in the first image (uv1) to
    lines in the second image (uv2).

    Since the fundamental matrix has arbitrary scale, it should
    be scaled such that ||f||_2 = 1.

    Args:
        uv1 (np.ndarray): image points in the first image (in pixels, Nx2)
        uv2 (np.ndarray): image points in the second image (in pixels, Nx2)

    Returns:
        np.ndarray: funadmental matrix (3x3)
    """

    assert uv1.shape[0] == uv2.shape[0]
    assert uv1.shape[1] == 2 and uv2.shape[1] == 2

    A = []
    ul = uv1[:, 0]
    vl = uv1[:, 1]

    ur = uv2[:, 0]
    vr = uv2[:, 1]

    for i in range(uv1.shape[0]):
        A.append([ur[i]*ul[i], ur[i]*vl[i], ur[i], vr[i]*ul[i], vr[i]*vl[i], vr[i], ul[i], vl[i], 1])
        

    A = np.array(A)
    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)
    
    norm_factor = np.linalg.norm(F)
    F = F/norm_factor
    return F

def point_to_epiline(pts1: np.ndarray, F: np.ndarray) -> np.ndarray:
    """
    Compute the epipolar line in the second image corresponding to each point
    in the first image.

    Args:
        pts1 (np.ndarray): points in the first image (Nx2)
        F (np.ndarray): fundamental matrix mapping points in the first image to
            lines in the second image (3x3)

    Returns:
        np.ndarray: lines in the second image (Nx3)
    """
    pts1_homogeneous = np.hstack((pts1, np.ones((pts1.shape[0], 1))))

    #  epipolar lines in the second image
    lines_in_second_image = np.dot(F, pts1_homogeneous.T).T

    return lines_in_second_image


def error_F(uv1: np.ndarray, uv2: np.ndarray, F: np.ndarray) -> np.ndarray:
    """
    Compute the mean distance from epiline to point on the second image.

    Args:
        uv1 (np.ndarray): image points in the first image (in pixels, Nx2)
        uv2 (np.ndarray): image points in the second image (in pixels, Nx2)
        F (np.ndarray): fundamental matrix mapping points in the first image to 
            lines in the second image (Nx3)

    Returns:
        np.ndarray: distances from epiline to point in the second image (Nx1).
    """

    epiline = point_to_epiline(uv1, F)
    uv2_homogeneous = np.hstack((uv2, np.ones((uv2.shape[0], 1))))

    distances = np.abs(np.sum(uv2_homogeneous * epiline, axis=1)) / np.sqrt(epiline[:, 0]**2 + epiline[:, 1]**2)
    

    return distances

def estimate_F_RANSAC(uv1: np.ndarray, 
                      uv2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate the fundamental matrix for the two camera views using RANSAC.
    Return the estimate of F, the list of inlier indices, and the best error
    computed using error_F().

    Args:
        uv1 (np.ndarray): image points in the first image (in pixels, Nx2)
        uv2 (np.ndarray): image points in the second image (in pixels, Nx2)

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Fundamental matrix, inlier 
            indices, and distance error over the inliers.
    """
    max_inliers = 0
    best_F = None
    best_inliers = []
    best_error = float('inf')  


    for i in range(ransac_n):
        idxs = np.random.choice(uv1.shape[0], 8, replace=False)
        rand_uv1 = uv1[idxs, :]
        rand_uv2 = uv2[idxs, :]

        F_est = estimate_F(rand_uv1, rand_uv2)
        dist = error_F(uv1, uv2, F_est)

        inliers = np.where(dist < ransac_eps)[0]

        if len(inliers) > max_inliers:
            max_inliers = len(inliers)
            best_F = F_est
            best_inliers = inliers
            best_error = dist

    return (best_F, best_inliers, best_error)

def estimate_E(F: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Compute the essential matrix from the fundamental matrix and intrinsics.

    Args:
        F (np.ndarray): fundamental matrix (3x3)
        K (np.ndarray): camera intrinsics (3x3)

    Returns:
        np.ndarray: essential matrix (3x3)
    """
    E = np.matmul(np.matmul(K.T, F), K)
    return E


def similarity_metric(window_left: np.ndarray, 
                      window_right: np.ndarray) -> float:
    """
    Compute the similarity measure between two image windows.

    Args:
        window_left (np.ndarray): left image window
        window_right (np.ndarray): right image window

    Returns:
        float: value of the similarity measure
    """
    assert window_left.shape == window_right.shape

    num = np.sum(window_left * window_right)
    left = np.sum(window_left**2)
    right = np.sum(window_right**2)
    denom = np.sqrt(left*right)
    return num/denom


def dense_point_correspondences(
    img_left: np.ndarray, 
    img_right: np.ndarray, 
    F: np.ndarray, 
    w: int,
    img_left_mask: np.ndarray) -> np.ndarray:
    """
    Find the coordinates of the point correspondence in img_right's frame for
    every point in img_left's frame.

    Args:
        img_left (np.ndarray): left image (HxW)
        img_right (np.ndarray): right image (HxW)
        F (np.ndarray): fundamental matrix mapping points in img_left to lines
            in img_right
        w (int): window size in pixels (odd number)
        img_left_mask (np.ndarray): boolean mask that is True in the foreground
            and False in the background. Only compute point correspondences in
            the foreground.

    Returns:
        np.ndarray: point correspondences in the right image for every point in 
            the left image (HxWx2)
    """

    height, width = img_left.shape
    assert img_left.shape == img_right.shape

    # Ensure window size is an odd number
    if w % 2 == 0:
        w += 1
    
    point_correspondences = np.zeros((height, width, 2), dtype=np.int32)

    #iterate through the left img 
    for r in range(0, height, 8):
        for c in range(0, width, 8):
            if img_left_mask[r, c]:
                # get left window 
                window_left = img_left[max(0, r - w//2):min(height, r + w//2 + 1),
                                       max(0, c - w//2):min(width, c + w//2 + 1)]

                # get the epipolar line in the right image
                line = point_to_epiline(np.array([[c, r]]), F)
                line = line[0]
                line_coordinates = utils.compute_line_coordinates(line, width)

                best_match = None
                best_score = -1

                # Iterate over the epipolar line 
                for rr, cr in line_coordinates:
                    # get window within the bounds of img dimensions 
                    if 0 <= rr < height and 0 <= cr < width:
                        # right image window
                        window_right = img_right[max(0, rr - w//2):min(height, rr + w//2 + 1),
                                                 max(0, cr - w//2):min(width, cr + w//2 + 1)]

                        if window_left.shape == window_right.shape:
                            #print(window_left.shape)
                            #print(window_right.shape)
                            # normalized cross-correlation
                            score = similarity_metric(window_left, window_right)

                            if score > best_score:
                                best_score = score
                                best_match = (rr, cr)
                                # print("BEST: ", best_match)
                # Update the point correspondences matrix
                point_correspondences[r, c] = best_match if best_match is not None else (0, 0)
    # plt.figure()

    # plt.imshow(point_correspondences[:,:,0])

    # plt.title("Row")

    # plt.show()

    # plt.figure()

    # plt.imshow(point_correspondences[:,:,1])

    # plt.title("Column")

    # plt.show()
    return point_correspondences

