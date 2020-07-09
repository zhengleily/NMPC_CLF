import numpy as np
import matplotlib.pyplot as plt
from quadrotor_dynamics import Quad
from quadenv import quad_env
from quadrotor_controller import quadrotor_controller, mpc_attitude_control
from matplotlib.colors import colorConverter
from matplotlib.animation import FuncAnimation
import mpl_toolkits.mplot3d as a3
import time
dt = 0.02
stime = 20
loop = int(stime / dt)

s = np.zeros(9)

xl = np.array([0, 0, 0])
vl = np.array([0, 0, 0])

xyTar = []
xyHis = []
uHis = []
tracking_err = []
time_con = []
xyTar_loop = []
xyHis_loop = []
uHis_loop = []
tracking_err_loop = []
u_old = np.zeros(4)

env = quad_env()

idx = np.arange(0, 20.02, dt)
course_pos = env.curve_pos(idx)
course_vel = env.curve_vel(idx)
# speed = 0.5  # m/s
# a = np.linalg.norm(course_vel, axis=0)
# course_vel = course_vel / np.linalg.norm(course_vel, axis=0) * (
#             speed)  # normalize with the same speed, but not direction.
course_vel[:, -1] = 0  # stop
course = np.concatenate((course_pos, course_vel), axis=0)  # desired state with dim 6, shape [dim, timeslot]

## Fixed Trajectory (point)
# xl = np.array([0, 0, 0.3])
# vl = np.array([0, 0, 0])

# horizon = 5 #for curved line
horizon = 20

# init state
s = env.reset()
#+ np.array([0.1, 0.2, 0., 0., 0., 0.])  # np.array([-1., 1., 0., 0., 0., 0.]) #
for t in range(loop):
    xl = course[:3, t: t + horizon]
    vl = course[3:, t: t + horizon]
    print("step: ", t)
    start_time = time.time()
    if xl.shape[1] < horizon:
        horizon = xl.shape[1]
    # u_pid = quadrotor_controller(s, xl, vl, 0)
    u_mpc, u_next = mpc_attitude_control(s, xl, vl, horizon, dt, u_old)
    end_time = time.time()

    u = u_mpc
    # u = u_pid
    # u_old = u_next
    print("state: ", s)
    print("control: ", u)
    # print("pid: ", u_pid)
    time_con.append(end_time - start_time)
    print("time: ", end_time - start_time)
    print("----------------------")
    s, _, _ = env.step(u)

    xyTar_loop.append(np.concatenate([xl[:,0], vl[:,0]]))
    xyHis_loop.append(s)
    uHis_loop.append(u)
    if t %100 == 10:
        a = np.vstack(xyHis_loop)
        tracking_err = course_pos[:,:t] - a[:t, 0:3].T
        xyTar = np.stack(xyTar_loop)
        xyHis = np.stack(xyHis_loop)
        uHis  = np.stack(uHis_loop)
        # time_consuming =

        plt.figure()
        plt.plot(time_con)
        plt.title("Time consuming")
        plt.show()

        plt.figure()
        plt.plot(uHis[:, 1:])
        plt.title("Angular velocity")
        plt.show()

        plt.figure()
        plt.plot(uHis[:, 0])
        plt.title("Thrust")
        plt.show()

        plt.figure()
        plt.plot(xyHis[:, 3:])
        plt.title("angle")
        plt.show()

        plt.figure()
        plt.plot(xyTar[:, 0], label='x_true')
        plt.plot(xyHis[:, 0], label='x_ref')
        plt.title("x")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(xyTar[:, 1], label='y_true')
        plt.plot(xyHis[:, 1], label='y_ref')
        plt.title("y")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(xyTar[:, 2], label='z_true')
        plt.plot(xyHis[:, 2], label='z_ref')
        plt.title("z")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(tracking_err[0, :], label='x_error')
        plt.plot(tracking_err[1, :], label='y_error')
        plt.plot(tracking_err[2, :], label='z_error')
        plt.title("Tracking error")
        plt.legend()
        plt.show()

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_title("target")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.plot(xyHis[:, 0], xyHis[:, 1], xyHis[:, 2], c='r', label='true')
        ax.plot(xyTar[:, 0], xyTar[:, 1], xyTar[:, 2], c='b', linestyle='--', label='ref')
        ax.scatter(xyHis[0, 0], xyHis[0, 1], xyHis[0, 2], 'x')
        ax.scatter(xyHis[-1, 0], xyHis[-1, 1], xyHis[-1, 2], '0')
        plt.legend()
        plt.show()