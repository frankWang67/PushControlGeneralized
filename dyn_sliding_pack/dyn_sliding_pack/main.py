import numpy as np
import dyn_sliding_pack
from matplotlib import pyplot as plt
from matplotlib import animation


def main(N=20, tf=2.5):
    # set values
    system = dyn_sliding_pack.params.DynPusherParameters()
    system.m = 0.1
    system.l = 0.2
    system.h = 0.01
    system.r = 0.02
    system.v_max = 1.0
    system.omega_max = 1.0
    system.a_max = 2.0
    system.beta_max = 1.0
    system.f_max = 0.6

    env = dyn_sliding_pack.params.DynPusherEnvironmentParameters()
    env.g = 9.81
    env.mu_g = 0.2
    env.mu_p = 0.1

    traj = dyn_sliding_pack.params.DynPusherTrajectoryParameters()
    traj.r0 = [0.0, 0.0, 0.0, 0.0]
    traj.rf = [0.1, 0.5, -0.1*np.pi, 0.0]
    traj.v0 = [0.0, 0.0, 0.0, 0.0]
    traj.vf = [0.0, 0.0, 0.0, 0.0]
    traj.tf_min = 2.0
    traj.tf_max = 3.0
    traj.gamma = 0.0

    pbm = dyn_sliding_pack.params.DynPusherProblem()
    pbm.system = system
    pbm.env = env
    pbm.traj = traj

    # dynamic model
    # md = dyn_sliding_pack.model.DynamicPusherModel(pbm)
    md = dyn_sliding_pack.model.DynamicFrictionlessPusherModel(pbm)

    # discretize
    opt = dyn_sliding_pack.disc.discretize(md.continuous_pbm, N)

    # initial guess
    x0 = []
    x_traj, u_traj, z_traj = dyn_sliding_pack.init.make_initialization_solutions(pbm, md, N)
    for i in range(N):
        x0 += x_traj[:, i].tolist()
    for i in range(N):
        x0 += u_traj[:, i].tolist()
    for i in range(N):
        x0 += z_traj[:, i].tolist()

    import pdb; pdb.set_trace()

    # solver
    sol, stats = dyn_sliding_pack.sol.solve(opt, solver_name='ipopt', x0=x0, tf=tf)
    x_sol = sol['x'][:N*md.x.shape[0]].toarray().reshape(N, md.x.shape[0])

    import pdb; pdb.set_trace()

    # plot data
    dyn_sliding_pack.vis.plot_variable_and_constraint(md.continuous_pbm, sol, N, tf)

    # plot animation
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    plt.gca().set_aspect('equal')
    plt.xlim([-0.1, 0.6])
    plt.ylim([-0.1, 0.6])
    animator = dyn_sliding_pack.vis.ResultAnimation(pbm)
    animator.set_patches_for_animation(ax, x_sol[0, :].flatten())
    anim = animation.FuncAnimation(
        fig,
        animator.update_for_animation,
        fargs=(ax, x_sol),
        frames=N,
        interval=10,
        blit=True,
        repeat=False,
    )
    plt.show()

if __name__ == '__main__':
    main(N=200, tf=3.0)
