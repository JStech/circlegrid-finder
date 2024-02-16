#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import cv2 as cv
from itertools import product
from math import sin, cos, tan, pi

#im_size = (600, 800)
#im_center = np.array([300, 400])
#
#img = np.zeros(im_size, dtype='uint8')
#
#dist_coeffs = np.array([2e-6, 1e-12])
#
#for (i, j) in product(range(im_size[0]), range(im_size[1])):
#    px = np.array([i, j])
#    r = np.linalg.norm(px - im_center)
#    c = dist_coeffs[0] * r**2 + dist_coeffs[1] * r**4
#    #px_d = (px + c * im_center) / (1 + c)
#    px_u = px + (px - im_center) * c
#    if int(px_u[0] / 100)%2 ^ int(px_u[1] / 100)%2:
#        img[i, j] = 255
#
#cv.imshow("", img)
#cv.waitKey(0)

asymm_grid_size = (8, 3)

asymm_grid = np.array(list(product(range(0, asymm_grid_size[0], 2), range(0, 2*asymm_grid_size[1], 2), [1])) +
                      list(product(range(1, asymm_grid_size[0], 2), range(1, 2*asymm_grid_size[1], 2), [1]))) - \
                              np.array([(asymm_grid_size[0]-1)/2, asymm_grid_size[1], 0])

H = np.array([[ 0.6,  0.0,  0.0],
              [ 0.0,  0.6,  0.0],
              [ 0.0,  0.1,  1.0]])
K1 = -1e-2
K2 = 1e-4
num_noise_pts = 20

grid_size = (4, 3)
grid = np.array(list(product(range(0, 2*grid_size[0], 2), range(0, 2*grid_size[1], 2), [1]))) - \
        np.array([grid_size[0]-1, grid_size[1]-1, 0])

theta = 0.1
offset = np.array([-0.4, 1.2, 0])
dgrid = asymm_grid @ H.T @ np.array([[cos(theta), sin(theta), 0], [-sin(theta), cos(theta), 0], [0, 0, 1]]) + offset
dgrid /= dgrid[:, 2:]

for i in range(dgrid.shape[0]):
    r = np.linalg.norm(dgrid[i, :2])
    dgrid[i, :2] *= 1 + K1 * r**2 + K2 * r**4

# add noise detections
dgrid = np.r_[dgrid, np.pad(10*np.random.random((num_noise_pts, 2))-5, ((0, 0), (0, 1)), constant_values=1)]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [3, 1, 1]})
ax1.plot(dgrid[:-num_noise_pts, 0], dgrid[:-num_noise_pts, 1], '.')
ax1.plot(dgrid[-num_noise_pts:, 0], dgrid[-num_noise_pts:, 1], '.')
ax1.set(xlim=[-5, 5], ylim=[-5, 5], aspect='equal')

#centroid = np.mean(dgrid, axis=0)
#c_idx = np.random.randint(dgrid.shape[0]) #np.argmin(np.linalg.norm(dgrid - centroid, axis=1))
#c = dgrid[c_idx, :]
#
#lines = []
#used = {c_idx}
#for i in range(dgrid.shape[0]):
#    if i in used: continue
#    line = np.cross(c, dgrid[i, :])
#    line /= np.linalg.norm(line[:2])
#    num_pts = 2
#    used.add(i)
#    for j in range(dgrid.shape[0]):
#        if j in (i, c_idx): continue
#        if abs(np.dot(line, dgrid[j, :])) <= 1:
#            used.add(j)
#            num_pts += 1
#    if num_pts >= min(asymm_grid_size[0]//2, asymm_grid_size[1]):
#        lines.append((num_pts, line))
#
#lines.sort(key=lambda l: l[0])
#left_edge = np.cross(np.array([-5, -5, 1]), np.array([-5, 5, 1]))
#right_edge = np.cross(np.array([5, -5, 1]), np.array([5, 5, 1]))
#for _, line in lines:
#    left_pt = np.cross(line, left_edge)
#    left_pt /= left_pt[2]
#    right_pt = np.cross(line, right_edge)
#    right_pt /= right_pt[2]
#    plt.plot([left_pt[0], right_pt[0]], [left_pt[1], right_pt[1]], "-")
#
#plt.show()

acc_size = (45, 200)

acc = np.zeros(acc_size)

blur = 5

for i in range(dgrid.shape[0]):
    pt = dgrid[i, :]
    for theta_i in range(acc_size[0]):
        theta = pi * theta_i / acc_size[0]
        rho = np.dot(pt[:2], [cos(theta), sin(theta)]) + 7
        rho_i = int((rho/14) * acc_size[1])
        for offset in range(-(blur//2), (blur//2)+1):
            if not 0 <= rho_i + offset < acc_size[1]: continue
            acc[theta_i, rho_i+offset] += 1

ax2.imshow(acc.T)

n_lines = 40
for i_line in range(n_lines):
    # get strongest line, zero out it and its neighbors
    (theta_i, rho_i) = np.unravel_index(acc.argmax(), acc.shape)
    for off_th, off_rh in product(range(-blur//2, blur//2 + 1), repeat=2):
        if not (0 <= rho_i + off_rh < acc_size[1] and 0 <= theta_i + off_th < acc_size[0]): continue
        acc[theta_i+off_th, rho_i+off_rh] = 0
    theta = pi * theta_i / acc_size[0]
    rho = rho_i/acc_size[1] * 14 - 7
    line = np.array([cos(theta), sin(theta), -rho])

    pts = []
    for i in range(dgrid.shape[0]):
        if abs(np.dot(line, dgrid[i, :])) < 1e-1:
            pts.append(dgrid[i, :])
    if len(pts) < 3: continue

    # re-fit line
    _, _, v = np.linalg.svd(np.stack(pts))
    line = v[-1, :]
    line /= np.linalg.norm(line[:2])

    # estimate vanishing point from each line

    # plot line
    if pi/4 < theta < 3*pi/4:
        pt1 = np.cross(line, np.array([1, 0, -5]))
        pt2 = np.cross(line, np.array([1, 0, 5]))
    else:
        pt1 = np.cross(line, np.array([0, 1, -5]))
        pt2 = np.cross(line, np.array([0, 1, 5]))
    pt1 /= pt1[2]
    pt2 /= pt2[2]
    ax1.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], '-', linewidth=0.5,
             color=colormaps['viridis'].colors[255 - 256*i_line//n_lines])

ax3.imshow(acc.T)
plt.show()
