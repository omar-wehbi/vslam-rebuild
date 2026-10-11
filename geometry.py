import numpy as np


def rt_to_t_inv(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.T
    T[:3, 3] = (-R.T @ t).flatten()
    return T


def world_to_camera_rt(world_t_cam):
    r_wc = world_t_cam[:3, :3]
    t_wc = world_t_cam[:3, 3]
    r_cw = r_wc.T
    t_cw = -r_cw @ t_wc
    return r_cw, t_cw


def pose_to_projection(world_t_cam, k):
    r_cw, t_cw = world_to_camera_rt(world_t_cam)
    return k @ np.hstack((r_cw, t_cw.reshape(3, 1)))


def camera_pose_to_projection(cam_t_world, k):
    return k @ cam_t_world[:3, :]


def project_point(proj, xw):
    x_h = np.array([xw[0], xw[1], xw[2], 1.0], dtype=np.float64)
    uvw = proj @ x_h
    return uvw[:2] / (uvw[2] + 1e-12)


# Backward compatibility for previous helper name.
Rt_to_T_inv = rt_to_t_inv
