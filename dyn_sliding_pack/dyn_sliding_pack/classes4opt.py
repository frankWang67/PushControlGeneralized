

class DynPusherParameters():
    m = None        # [kg] Slider mass
    l = None        # [m] Pusher board width
    h = None        # [m] Pusher board thickness
    r = None        # [m] Ball radius
    v_max = None    # [m/s] Maximum pusher linear velocity
    omega_max = None# [rad/s] Maximum pusher angular velocity
    a_max = None    # [m/s^2] Maximum pusher linear acceleration
    beta_max = None # [rad/s^2] Maximum pusher angular acceleration
    f_max = None    # [N] Maximum contact force

class DynPusherEnvironmentParameters():
    mu_g = None     # friction coefficient between ground and slider
    mu_p = None     # friction coefficient between pusher and slider
    g = None        # [m/s^2] Gravity vector

class DynPusherTrajectoryParameters():
    r0 = None       # Initial position [x_B, y_B, theta_B, x_M]
    rf = None       # Terminal position
    v0 = None       # Initial velocity
    vf = None       # Terminal velocity
    tf_min = None   # Minimum flight time
    tf_max = None   # Maximum flight time
    gamma = None    # Minimum-time vs. minimum-energy tradeoff

class DynPusherOptimizationConfig():
    ## sysbols
    x = None        # state vector
    u = None        # input vector
    p = None        # parameter vector
    z = None        # auxiliary & slack vector

    ## constraints
    f = None        # System dynamics (̇x = f(x(t),u(t)))
    g = None        # Path inequality constraints (g(x(t),u(t)) ≤ 0)
    h = None        # Path equality constraints (g(x(t),u(t)) = 0)
    hi = None       # Initial condition constraints (hi(x(t),u(t)) = 0)
    ht = None       # Terminal condition constraints (ht(x(t),u(t)) = 0)

    # auxiliary functions (for debug)
    aux_f_ground = None   # ground friction in board frame
    aux_f_inertia = None  # inertia force in board frame
    
    ## costs
    phi = None      # Terminal cost (ϕ(x(t),u(t)))
    gamma = None    # Path integral cost (Γ(x(t),u(t)))

    ## boundaries
    lbx = None      # lower bound on state vector
    ubx = None      # upper bound on state vector
    lbu = None      # lower bound on control vector
    ubu = None      # upper bound on control vector
    lbp = None      # lower bound on parameter vector
    ubp = None      # upper bound on parameter vector
    lbz = None      # lower bound on auxiliary vector
    ubz = None      # upper bound on auxiliary vector

class CasADiNLPOptions():
    f = None        # objective function
    x = None        # variables
    p = None        # objective function
    g = None        # constraints
    
    x0 = None       # initial guess
    lbx = None      # lower bound of x
    ubx = None      # upper bound of x
    lbg = None      # lower bound of g
    ubg = None      # upper bound of g
    discrete = None # whether or not the variable is an integer

    cfg = None      # configuration dict for nlp solvers 

class DynPusherProblem():
    system:DynPusherParameters = None           # system parameters
    env:DynPusherEnvironmentParameters = None   # environment parameters
    traj:DynPusherTrajectoryParameters = None   # trajectory parameters
