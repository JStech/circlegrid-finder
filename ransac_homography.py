#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import cv2 as cv
from itertools import product, combinations
from math import sin, cos, tan, pi, log
from collections import defaultdict

def gen_grid_pts(symmetric, grid_size):
    """Makes grid according to specifications, centered at the origin with largest dimension approximately filling [-1, 1]"""
    if symmetric:
        grid = np.array(list(product(range(0, 2*grid_size[0], 2), range(0, 2*grid_size[1], 2), [1]))) - \
                np.array([grid_size[0]-1, grid_size[1]-1, 0], dtype=float)
        grid[:, :2] /= (max(grid_size)-1)
        return grid
    else:
        grid = np.array(list(product(range(0, grid_size[0], 2), range(0, 2*grid_size[1], 2), [1])) +
                        list(product(range(1, grid_size[0], 2), range(1, 2*grid_size[1], 2), [1]))) - \
                                np.array([(grid_size[0]-1)/2, grid_size[1], 0], dtype=float)
        grid[:, :2] /= (max(grid_size[0]/2, grid_size[1])-1)
        return grid

def distort_grid(grid, H, K1, K2):
    """Apply homography (rotation/perspective) and radial distortion to grid points"""

    grid = grid @ H.T
    grid /= grid[:, 2:]

    for i in range(grid.shape[0]):
        r = np.linalg.norm(grid[i, :2])
        grid[i, :2] *= 1 + K1 * r**2 + K2 * r**4

    return grid

def add_noise_points(grid, num_points, limits, exclude_dist=None):
    """add extra false detections"""
    x_range = limits[0][1] - limits[0][0]
    y_range = limits[1][1] - limits[1][0]
    origin = np.array([limits[0][0], limits[1][0]])
    noise_pts = np.random.random((num_points, 2)) * np.array([x_range, y_range]) + origin
    if exclude_dist is not None:
        for _ in range(100):
            dists = np.min(np.linalg.norm(np.expand_dims(noise_pts, 0) - np.expand_dims(grid[:, :2], 1), axis=2), axis=0)
            assert dists.shape == (num_points,)
            keep = dists > exclude_dist
            if np.sum(keep) == num_points: break
            new_pts = np.random.random((num_points - np.sum(keep), 2)) * np.array([x_range, y_range]) + origin
            noise_pts = np.r_[noise_pts[keep, :], new_pts]
    return np.r_[grid, np.pad(noise_pts, ((0, 0), (0, 1)), constant_values=1)]

def add_detection_noise(grid, sigma):
    """add zero-mean normally distributed isotropic noise to detections"""
    grid[:, :2] += sigma * np.random.randn(grid.shape[0], 2)
    return grid

def estimate_homography(pts):
    """estimate homography from four points in + pattern"""
    A = np.zeros((8, 9))
    pts_plane = np.array([
        [1, 0, 1],
        [-1, 0, 1],
        [0, 1, 1],
        [0, -1, 1],
        ])
    for i, (pt, ppl) in enumerate(zip(pts, pts_plane)):
        A[2*i:2*i+2, :] = np.r_[
                [np.r_[0, 0, 0, -ppl[2] * pt, ppl[1] * pt]],
                [np.r_[ppl[2] * pt, 0, 0, 0, -ppl[0] * pt]],
                ]
    _, _, v = np.linalg.svd(A)
    return v[-1, :].reshape(3, 3)

def check_line(pts_rounded, line_ids):
    """Verify that pts_rounded specified in line_ids are in a line, with -1 indicating missing points"""
    if np.sum(line_ids != -1) < 3:
        return True
    idx, pts = list(zip(*[(i, pts_rounded[pt_id]) for i, pt_id in enumerate(line_ids) if pt_id != -1]))
    v = (pts[1] - pts[0])/(idx[1] - idx[0])
    for i in range(len(idx) - 1):
        if not np.allclose(pts[i+1] - pts[i], v * (idx[1] - idx[0])):
            print(i, pts[i+1], pts[i], v, idx[i + 1], idx[i])
            return False
    return True


# check that lines make lines
def check_grid(pts_rounded, grid_ids, grid_size, symmetric):
    print("checking")
    print(grid_ids)
    """Check that points for lines along rows and columns of grid"""
    for i in range(grid_size[0]):
        if not check_line(pts_rounded, grid_ids[i, :]):
            print("exit 1")
            return False
    for j in range(0, grid_size[1]):
        if symmetric:
            if not check_line(pts_rounded, grid_ids[:, j]):
                print("exit 2")
                return False
        else:
            if not check_line(pts_rounded, grid_ids[::2, j]):
                print("exit 3")
                return False
            if not check_line(pts_rounded, grid_ids[1::2, j]):
                print("exit 4", j, grid_ids[1::2, j], pts_rounded[grid_ids[1::2, j], :])
                return False
    print("OK")
    return True

def find_grid_points(pts, seed, grid_size, symmetric):
    """Identify which points in pts are near integer coordinates"""
    # round
    pts_rounded = np.rint(pts)
    bbox = [np.min(pts_rounded[:, :2], axis=0), np.max(pts_rounded[:, :2])]

    # reject outliers
    outliers = set()
    for i, j in combinations(range(pts.shape[0]), 2):
        if np.linalg.norm(pts_rounded[i, :] - pts_rounded[j, :]) < 1e-6:
            int_dist_i = np.linalg.norm(pts_rounded[i, :] - pts[i, :])
            int_dist_j = np.linalg.norm(pts_rounded[j, :] - pts[j, :])
            outliers.add(i if int_dist_i > int_dist_j else j)

    for i in range(pts.shape[0]):
        neighbors = 0
        for v in (np.array([1, 0, 0]), np.array([-1, 0, 0]), np.array([0, 1, 0]), np.array([0, -1, 0])):
            neighbors += (np.min(np.linalg.norm(pts_rounded - (pts_rounded[i, :] + v), axis=1)) < 1e-6)
        if neighbors == 0:
            outliers.add(i)

    inliers = set(range(pts.shape[0])) - outliers
    if len(inliers) < grid_size[0] * grid_size[1]:
        return

    # lines that might align with grid axes
    axis_lines = [
            np.array([ 0,  1,  0]),
            np.array([ 1,  0,  0]),
            np.array([ 1,  1,  0]),
            np.array([ 1, -1,  0]),
            np.array([ 1,  2,  0]),
            np.array([ 1, -2,  0]),
            np.array([ 2,  1,  0]),
            np.array([ 2, -1,  0]),
            ]

    residual_distributions = []
    x_axis_limits = []
    y_axis_limits = []
    for axis_line in axis_lines:
        distribution = defaultdict(int)
        for pt in pts_rounded[list(inliers), :]:
            distribution[int(np.dot(axis_line, pt))] += 1
        residual_distributions.append(distribution)

        # if this in fact is the x axis, which lines (residuals) have the correct number of points?
        x_enough_pts = [r for r, c in distribution.items() if c >= grid_size[0] // (2 - symmetric)]
        if len(x_enough_pts) >= grid_size[1] * (2 - symmetric) and 0 in x_enough_pts:
            x_axis_limits.append((min(x_enough_pts), max(x_enough_pts)))
        else:
            x_axis_limits.append(None)
        y_enough_pts = [r for r, c in distribution.items() if c >= grid_size[1]]
        if len(y_enough_pts) >= grid_size[0] and 0 in y_enough_pts:
            y_axis_limits.append((min(y_enough_pts), max(y_enough_pts)))
        else:
            y_axis_limits.append(None)

    grid_ids = -1 * np.ones(grid_size, dtype=int)

    # usually just one pair of axes meets the above criteria
    for i_x, i_y in product(range(len(axis_lines)), repeat=2):
        if i_x == i_y or x_axis_limits[i_x] is None or y_axis_limits[i_y] is None: continue

        x_axis = axis_lines[i_x]
        y_axis = axis_lines[i_y]
        for i, j in product(range(grid_size[0]), range(grid_size[1])):
            if symmetric:
                pt = np.cross(
                        np.r_[x_axis[:2], -(x_axis_limits[i_x][0] + j)],
                        np.r_[y_axis[:2], -(y_axis_limits[i_y][0] + i)])
            else:
                pt = np.cross(
                        np.r_[x_axis[:2], -(x_axis_limits[i_x][0] + 2*j + i%2)],
                        np.r_[y_axis[:2], -(y_axis_limits[i_y][0] + i)])
            pt = (pt / pt[2]).astype(int)
            grid_ids[i, j] = np.argmin(np.linalg.norm(pts_rounded - pt, axis=1))

    return grid_ids

def ransac_homography(pts, grid_size, symmetric):
    """find + pattern to recover homography"""

    expected_num_pts = grid_size[0] * grid_size[1]
    cos_thresh = 0.99
    found = False
    seeds = list(range(pts.shape[0]))
    while len(seeds) > 0:
        seed_i = np.random.randint(len(seeds))
        print("trying ", seed_i)
        seed = seeds.pop(seed_i)
        pt = pts[seed, :]
        dists = np.linalg.norm(pts - pt, axis=1)
        nearest_neighbors = np.argsort(dists)
        assert dists[nearest_neighbors[0]] < 1e-9, f"{nearest_neighbors}"

        for n_plus in combinations(nearest_neighbors[:10], 4):
            n_pts = pts[n_plus, :]
            intersection = np.cross(np.cross(n_pts[0], n_pts[1]), np.cross(n_pts[2], n_pts[3]))
            intersection /= intersection[2]
            if np.linalg.norm(intersection - pt) > 1e-2:
                continue

            h = estimate_homography(n_pts)

            pts_plane = pts @ h.T
            pts_plane /= pts_plane[:, 2:]

            # first check: does this map the seed to the origin?
            if np.linalg.norm(pts_plane[seed, :2]) > 1e-2: continue

            grid_ids = find_grid_points(pts_plane, seed, grid_size, symmetric)
            if grid_ids is None: continue

            plt.subplot(1, 2, 1)
            plt.plot(pts[:expected_num_pts, 0], pts[:expected_num_pts, 1], '.')
            plt.plot(pts[expected_num_pts:, 0], pts[expected_num_pts:, 1], '.')
            plt.plot(pts[n_plus, 0], pts[n_plus, 1], 'o')
            plt.plot(pts[grid_ids.flat, 0], pts[grid_ids.flat, 1], ':')
            plt.gca().set_aspect('equal')
            plt.subplot(1, 2, 2)
            #plt.plot(pts_plane[list(inliers), 0], pts_plane[list(inliers), 1], '.')
            #pp_rounded = np.rint(pts_plane)
            #plt.plot(pp_rounded[list(inliers), 0], pp_rounded[list(inliers), 1], '.')
            plt.plot(pts_plane[:, 0], pts_plane[:, 1], '.')
            plt.plot(pts_plane[grid_ids.flat, 0], pts_plane[grid_ids.flat, 1], ':')
            plt.gca().set_aspect('equal')
            plt.gca().set_xticks(range(*map(int, plt.gca().get_xlim())))
            plt.gca().set_yticks(range(*map(int, plt.gca().get_ylim())))
            plt.gca().grid()
            plt.show()
            return grid_ids

if __name__ == "__main__":

    asymm_grid_size = (10, 4)
    symm_grid_size = (5, 4)

    H = np.array([[ 0.6 + 0.2*np.random.randn(),  0.1*np.random.randn(), 0.5*np.random.randn()],
                  [ 0.1*np.random.randn(),  0.5 + 0.3*np.random.randn(),  0.5*np.random.randn()],
                  [ 0.1*np.random.randn(),  0.1*np.random.randn(),  1.0]])
    if np.linalg.det(H) < 0:
        H[:, 0] *= -1
    theta = np.random.randn()
    H @= np.array([[cos(theta), sin(theta), 0], [-sin(theta), cos(theta), 0], [0, 0, 1]])

    K1 = -1e-2
    K2 = 5e-5

    num_noise_pts = 20
    noise_limit = 2
    ex_dist = 0.01
    sigma = 5e-5

    symm_grid = add_detection_noise(
            distort_grid(
                add_noise_points(
                    gen_grid_pts(True, symm_grid_size), num_noise_pts,
                    ((-noise_limit, noise_limit), (-noise_limit, noise_limit)),
                    ex_dist
                    ), H, K1, K2), sigma)
    asymm_grid = add_detection_noise(
            distort_grid(
                add_noise_points(
                    gen_grid_pts(False, asymm_grid_size), num_noise_pts,
                    ((-noise_limit, noise_limit), (-noise_limit, noise_limit)),
                    ex_dist
                    ), H, K1, K2), sigma)

    ransac_homography(asymm_grid, asymm_grid_size, False)
    ransac_homography(symm_grid, symm_grid_size, True)
    plt.subplot(1, 2, 1)
    plt.plot(symm_grid[:-num_noise_pts, 0], symm_grid[:-num_noise_pts, 1], 'o')
    plt.plot(symm_grid[-num_noise_pts:, 0], symm_grid[-num_noise_pts:, 1], 'o')
    for i, pt in enumerate(symm_grid):
        plt.gca().annotate(str(i), pt[:2], fontsize="x-small")
    #xlim = plt.gca().get_xlim()
    #ylim = plt.gca().get_ylim()
    #left_edge = np.array([1, 0, -xlim[0]])
    #right_edge = np.array([1, 0, -xlim[1]])
    #for line in symm_lines.items():
    #    left_pt = np.cross(left_edge, line[1])
    #    left_pt /= left_pt[2]
    #    right_pt = np.cross(right_edge, line[1])
    #    right_pt /= right_pt[2]
    #    plt.plot([left_pt[0], right_pt[0]], [left_pt[1], right_pt[1]], linewidth=0.5)
    #plt.gca().set_xlim(xlim)
    #plt.gca().set_ylim(ylim)
    plt.gca().set_aspect('equal')
    plt.subplot(1, 2, 2)
    plt.plot(asymm_grid[:-num_noise_pts, 0], asymm_grid[:-num_noise_pts, 1], 'o')
    plt.plot(asymm_grid[-num_noise_pts:, 0], asymm_grid[-num_noise_pts:, 1], 'o')
    for i, pt in enumerate(asymm_grid):
        plt.gca().annotate(str(i), pt[:2], fontsize="x-small")
    #xlim = plt.gca().get_xlim()
    #ylim = plt.gca().get_ylim()
    #left_edge = np.array([1, 0, -xlim[0]])
    #right_edge = np.array([1, 0, -xlim[1]])
    #for line in asymm_lines.items():
    #    left_pt = np.cross(left_edge, line[1])
    #    left_pt /= left_pt[2]
    #    right_pt = np.cross(right_edge, line[1])
    #    right_pt /= right_pt[2]
    #    plt.plot([left_pt[0], right_pt[0]], [left_pt[1], right_pt[1]], linewidth=0.5)
    #plt.gca().set_xlim(xlim)
    #plt.gca().set_ylim(ylim)
    plt.gca().set_aspect('equal')
    plt.show()
