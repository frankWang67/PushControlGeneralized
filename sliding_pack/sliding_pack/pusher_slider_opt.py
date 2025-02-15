# Author: Joao Moura
# Contact: jpousad@ed.ac.uk
# Date: 02/06/2021
# -------------------------------------------------------------------
# Description:
# 
# Class for the trajectory optimization (TO) for the pusher-slider 
# problem using a Non-Linear Program (NLP) approach
# -------------------------------------------------------------------

# import libraries
import sys
import os
import time
import numpy as np
import casadi as cs
import sliding_pack

class buildOptObj():

    def __init__(self, dyn_class, timeHorizon, configDict, psic_offset=0.0, X_nom_val=None,
                 U_nom_val=None, dt=0.1, useGoalFlag=False, max_iter=None):

        # init parameters
        print('**** Initialize the optimization problem ****')

        self.dyn = dyn_class
        self.TH = timeHorizon
        self.solver_name = configDict['solverName']
        self.W_x = cs.diag(cs.SX(configDict['W_x']))
        self.W_u = cs.diag(cs.SX(configDict['W_u']))[:self.dyn.Nu,
                                                     :self.dyn.Nu]
        self.K_goal = configDict['K_goal']
        self.numObs = configDict['numObs']
        if X_nom_val is None:
            self.X_nom_val = cs.DM.zeros(self.dyn.Nx, self.TH)
        else:
            self.X_nom_val = X_nom_val
        if U_nom_val is None:
            self.U_nom_val = cs.DM.zeros(self.dyn.Nu, self.TH-1)
        else:
            self.U_nom_val = U_nom_val
        self.useGoalFlag = useGoalFlag
        self.solverName = configDict['solverName']
        self.linDyn = configDict['linDynFlag']
        self.code_gen = configDict['codeGenFlag']
        self.no_printing = configDict['noPrintingFlag']
        self.phases = configDict['phases']

        self.x_bias = cs.SX.zeros(self.dyn.Nx)
        self.x_bias[-1] = psic_offset

        # opt var dimensionality
        self.Nxu = self.dyn.Nx + self.dyn.Nu
        self.Nopt = self.Nxu + self.dyn.Nz

        # initialize variables for opt and args
        self.opt = sliding_pack.opt.OptVars()
        self.opt.x = []
        self.opt.g = []
        self.opt.f = []
        self.opt.p = []
        self.opt.discrete = []
        self.args = sliding_pack.opt.OptArgs()
        self.args.lbx = []
        self.args.ubx = []
        self.args.lbg = []
        self.args.ubg = []

        # disturbance observer
        self.d_hat = cs.SX.sym('d_hat', self.dyn.Nx)

        # set optimization variables
        self.X_nom = cs.SX.sym('X_nom', self.dyn.Nx, self.TH)
        if self.linDyn:
            self.U_nom = cs.SX.sym('U_nom', self.dyn.Nu, self.TH-1)
            # define vars for deviation from nominal path
            self.X_bar = cs.SX.sym('X_bar', self.dyn.Nx, self.TH)
            self.U_bar = cs.SX.sym('U_bar', self.dyn.Nu, self.TH-1)
            # define path variables
            self.X = self.X_nom + self.X_bar
            self.U = self.U_nom + self.U_bar
        else:
            # define path variables
            self.X = cs.SX.sym('X', self.dyn.Nx, self.TH)
            self.U = cs.SX.sym('U', self.dyn.Nu, self.TH-1)
            # define vars for deviation from nominal path
            self.X_bar = self.X - self.X_nom
            # normalize angles
            # self.X_bar[2, :] = cs.fmod(cs.fabs(self.X_bar[2, :]), 2*cs.pi) - cs.pi
        # initial state
        self.x0 = cs.SX.sym('x0', self.dyn.Nx)
        if self.phases is None:
            self.Nphases = self.TH-1
            self.Zphases = cs.SX.sym('Z', self.dyn.Nz, self.Nphases)
            self.Z = self.Zphases
        else:
            if np.sum(self.phases) != self.TH-1:
                print('Error: Number of steps {} in phases does not match time horizon {}-1.'.format(np.sum(self.phases), self.TH)) 
                sys.exit()
            else:
                self.Nphases = len(self.phases)
                self.Zphases = cs.SX.sym('Z', self.dyn.Nz, self.Nphases)
                self.Z = cs.repmat(self.Zphases[:, 0], 1, self.phases[0])
                for i in range(1, self.Nphases):
                    self.Z = cs.horzcat(self.Z, cs.repmat(self.Zphases[:, i], 1, self.phases[i]))
        self.Nzvars = self.dyn.Nz * self.Nphases

        # constraint functions
        #  -------------------------------------------------------------------
        # ---- Define Dynamic constraints ----
        print("**** Building the optimization problem ****")

        __x_bar = cs.SX.sym('x_bar', self.dyn.Nx)
        if self.linDyn:
            # define gradients of the dynamic
            __u_nom = cs.SX.sym('u_nom', self.dyn.Nu)
            __x_nom = cs.SX.sym('x_nom', self.dyn.Nx)
            __A_func = cs.Function(
                    'A_func', [__x_nom, __u_nom, self.dyn.beta],
                    [cs.jacobian(self.dyn.f(__x_nom, __u_nom, self.dyn.beta), __x_nom)],
                    ['x', 'u', 'beta'], ['A'])
            __B_func = cs.Function(
                    'B_func', [__x_nom, __u_nom, self.dyn.beta],
                    [cs.jacobian(self.dyn.f(__x_nom, __u_nom, self.dyn.beta), __u_nom)],
                    ['x', 'u', 'beta'], ['B'])
            # define dynamics error
            __x_bar_next = cs.SX.sym('x_bar_next', self.dyn.Nx)
            __u_bar = cs.SX.sym('u_bar', self.dyn.Nu)
            self.f_error = cs.Function(
                    'f_error',
                    [__x_nom, __u_nom, __x_bar, __x_bar_next, __u_bar, self.dyn.beta],
                    [__x_bar_next-__x_bar-dt*(cs.mtimes(__A_func(__x_nom, __u_nom, self.dyn.beta), __x_bar) + cs.mtimes(__B_func(__x_nom,__u_nom, self.dyn.beta),__u_bar))])
        else:
            __x_next = cs.SX.sym('__x_next', self.dyn.Nx)
            self.f_error = cs.Function(
                    'f_error',
                    [self.dyn.x, self.dyn.u, __x_next, self.dyn.beta, self.d_hat],
                    [__x_next-self.dyn.x-dt*(self.dyn.f(self.dyn.x,self.dyn.u,self.dyn.beta)+self.d_hat)])
        # ---- Map dynamics constraint ----
        self.F_error = self.f_error.map(self.TH-1)
        #  -------------------------------------------------------------------
        # control constraints
        self.G_u = self.dyn.g_u.map(self.TH-1)
        #  -------------------------------------------------------------------)

        #  -------------------------------------------------------------------
        self.cost_f = cs.Function(
                'cost_f',
                [__x_bar, self.dyn.u],
                [cs.dot(__x_bar, cs.mtimes(self.W_x, __x_bar))
                    + cs.dot(self.dyn.u, cs.mtimes(self.W_u, self.dyn.u))])
        self.cost_F = self.cost_f.map(self.TH-1)
        # ------------------------------------------
        if self.dyn.Nz > 0:
            self.kz_F = self.dyn.kz_f.map(self.TH-1)
            xz = np.linspace(0, 1, self.TH-1)
            self.Kz = self.kz_F(xz).T
        #  -------------------------------------------------------------------

        #  -------------------------------------------------------------------
        #  Building the Problem
        #  -------------------------------------------------------------------

        # ---- Set optimization variables ----
        print('**** Set optimization variables ****')

        if self.linDyn:
            for i in range(self.TH-1):
                # ---- Add States to optimization variables ---
                self.opt.x += self.X_bar[:, i].elements()
                self.args.lbx += [-cs.inf]*self.dyn.Nx
                self.args.ubx += [cs.inf]*self.dyn.Nx
                self.opt.discrete += [False]*self.dyn.Nx
                # ---- Add Actions to optimization variables ---
                self.opt.x += self.U_bar[:, i].elements()
                self.args.lbx += [-cs.inf]*self.dyn.Nu
                self.args.ubx += [cs.inf]*self.dyn.Nu
                self.opt.discrete += [False]*self.dyn.Nu
            self.opt.x += self.X_bar[:, -1].elements()
            self.args.lbx += [-cs.inf]*self.dyn.Nx
            self.args.ubx += [cs.inf]*self.dyn.Nx
            self.opt.discrete += [False]*self.dyn.Nx
        else:
            for i in range(self.TH-1):
                # ---- Add States to optimization variables ---
                self.opt.x += self.X[:, i].elements()
                if i == 0: # expanding state constraint for 1st one (what trick is this?)
                    self.args.lbx += [1.5*x for x in self.dyn.lbx]
                    self.args.ubx += [1.5*x for x in self.dyn.ubx]
                else:
                    self.args.lbx += self.dyn.lbx
                    self.args.ubx += self.dyn.ubx
                self.opt.discrete += [False]*self.dyn.Nx
                # ---- Add Actions to optimization variables ---
                self.opt.x += self.U[:, i].elements()
                self.args.lbx += self.dyn.lbu
                self.args.ubx += self.dyn.ubu
                self.opt.discrete += [False]*self.dyn.Nu
            self.opt.x += self.X[:, -1].elements()
            self.args.lbx += self.dyn.lbx
            self.args.ubx += self.dyn.ubx
            self.opt.discrete += [False]*self.dyn.Nx
        for i in range(self.Nphases):
            # ---- Add slack/additional opt variables ---
            self.opt.x += self.Zphases[:, i].elements()
            self.args.lbx += self.dyn.lbz
            self.args.ubx += self.dyn.ubz
            self.opt.discrete += [self.dyn.z_discrete]*self.dyn.Nz

        # ---- Set optimzation constraints ----
        print('**** Set optimization constraints ****')

        self.opt.g = (self.X[:, 0]-self.x0).elements()  # Initial Conditions
        self.args.lbg = [0.0]*self.dyn.Nx
        self.args.ubg = [0.0]*self.dyn.Nx
        # ---- Dynamic constraints ---- 
        print("-- Dynamic constraints --")
        if self.linDyn:
            print("F_error calculation started")
            start_time = time.time()
            self.opt.g += self.F_error(
                    self.X_nom[:, :-1] + self.x_bias, self.U_nom,
                    self.X_bar[:, :-1] + self.x_bias, self.X_bar[:, 1:] + self.x_bias,
                    self.U_bar,
                    self.dyn.beta).elements()
            print(f"F_error calculation ended, time taken: {time.time()-start_time} seconds")
        else:
            print("F_error calculation started")
            start_time = time.time()
            self.opt.g += self.F_error(
                    self.X[:, :-1] + self.x_bias, self.U, 
                    self.X[:, 1:] + self.x_bias,
                    self.dyn.beta,
                    self.d_hat).elements()
            print(f"F_error calculation ended, time taken: {time.time()-start_time} seconds")
        self.args.lbg += [0.] * self.dyn.Nx * (self.TH-1)
        self.args.ubg += [0.] * self.dyn.Nx * (self.TH-1)
        # ---- Friction constraints ----
        print("-- Friction constraints --")
        self.opt.g += self.G_u(self.U, self.Z).elements()
        self.args.lbg += self.dyn.g_lb * (self.TH-1)
        self.args.ubg += self.dyn.g_ub * (self.TH-1)
        if self.linDyn:
            # ---- Action constraints
            print("-- Action constraints --")
            for i in range(self.TH-1):
                self.opt.g += self.U[:, i].elements()
                self.args.lbg += self.dyn.lbu
                self.args.ubg += self.dyn.ubu

        # ---- Add constraints for obstacle avoidance ----
        print("-- Obstacle avoidance constraints --")
        if self.numObs > 0:
            obsC = cs.SX.sym('obsC', 2, self.numObs)
            obsR = cs.SX.sym('obsR', self.numObs)
            for i_obs in range(self.numObs):
                for i_th in range(self.TH-1):
                    self.opt.g += (cs.norm_2(self.dyn.s(self.X[:, i_th+1])[:2]-obsC[:, i_obs])**2 - (obsR[i_obs]+self.dyn.Radius)**2).elements()
                    self.args.lbg += [0.]
                    self.args.ubg += [cs.inf]

        # ---- optimization cost ----
        print('**** Set optimization cost ****')
        if self.useGoalFlag:
            self.S_goal = cs.SX.sym('s', self.TH-1)
            self.X_goal_var = cs.SX.sym('x_goal_var', self.dyn.Nx)
            self.opt.f = self.cost_f(cs.mtimes(self.X[:, :-1], self.S_goal) - self.X_goal_var, self.U[:, -1])
        else:
            self.opt.f = cs.sum2(self.cost_F(self.X_bar[:, :-1], self.U))
            self.opt.f += self.K_goal*self.cost_f(self.X_bar[:, -1], self.U[:, -1])
        for i in range(self.dyn.Nz):
            # (what trick is this?)
            self.opt.f += cs.sum1(self.Kz*(self.Z[i].T**2))

        # ---- Set optimization parameters ----
        print('**** Set optimization parameters ****')

        self.opt.p = []
        self.opt.p += self.dyn.beta.elements()
        self.opt.p += self.x0.elements()
        self.opt.p += self.X_nom.elements()
        if not self.linDyn:
            self.opt.p += self.d_hat.elements()
        if self.useGoalFlag:
            self.opt.p += self.X_goal_var.elements()
            self.opt.p += self.S_goal.elements()
        if self.linDyn:
            self.opt.p += self.U_nom.elements()
        if self.numObs > 0:
            for i_obs in range(self.numObs):
                self.opt.p += obsC[:, i_obs].elements()
                self.opt.p += obsR[i_obs].elements()

        # Set up QP Optimization Problem
        #  -------------------------------------------------------------------
        # ---- Set solver options ----
        opts_dict = {'print_time': 0}
        prog_name = 'MPC' + '_TH' + str(self.TH) + '_' + self.solver_name + '_codeGen_' + str(self.code_gen)
        if self.solver_name == 'ipopt':
            if self.no_printing: opts_dict['ipopt.print_level'] = 0
            if max_iter is not None:
                opts_dict['ipopt.max_iter'] = max_iter
            opts_dict['ipopt.jac_d_constant'] = 'yes'
            opts_dict['ipopt.warm_start_init_point'] = 'yes'
            opts_dict['ipopt.hessian_constant'] = 'yes'
        if self.solver_name == 'knitro':
            opts_dict['knitro'] = {}
            opts_dict['knitro.outlev'] = 0
            # opts_dict['knitro.maxit'] = 80
            opts_dict['knitro.feastol'] = 1.e-3
            if self.no_printing: opts_dict['knitro']['mip_outlevel'] = 0
        if self.solver_name == 'snopt':
            opts_dict['snopt'] = {}
            if self.no_printing: opts_dict['snopt'] = {'Major print level': '0', 'Minor print level': '0'}
            opts_dict['snopt']['Hessian updates'] = 1
        if self.solver_name == 'qpoases':
            if self.no_printing: opts_dict['printLevel'] = 'none'
            opts_dict['sparse'] = True
        if self.solver_name == 'gurobi':
            if self.no_printing: opts_dict['gurobi.OutputFlag'] = 0
        # ---- Create solver ----
        print('**** Create solver *****')
        print(len(self.opt.x))
        print(len(self.opt.g))
        print(len(self.opt.p))
        print('************************')
        prob = {'f': self.opt.f,
                'x': cs.vertcat(*self.opt.x),
                'g': cs.vertcat(*self.opt.g),
                'p': cs.vertcat(*self.opt.p)
                }
        # ---- add discrete flag ----
        opts_dict['discrete'] = self.opt.discrete  # add integer variables
        # ---- fix the NaN bug ----
        opts_dict['calc_lam_p'] = False
        if (self.solver_name == 'ipopt') or (self.solver_name == 'snopt') or (self.solver_name == 'knitro'):
            self.solver = cs.nlpsol('solver', self.solver_name, prob, opts_dict)
            if self.code_gen:
                if not os.path.isfile('./' + prog_name + '.so'):
                    self.solver.generate_dependencies(prog_name + '.c')
                    os.system('gcc -fPIC -shared -O3 ' + prog_name + '.c -o ' + prog_name + '.so')
                self.solver = cs.nlpsol('solver', self.solver_name, prog_name + '.so', opts_dict)
        elif (self.solver_name == 'gurobi') or (self.solver_name == 'qpoases'):
            self.solver = cs.qpsol('solver', self.solver_name, prob, opts_dict)
        #  -------------------------------------------------------------------

    def solveProblem(self, idx, x0, beta, d_hat=None,
                     X_warmStart=None, U_warmStart=None,
                     obsCentre=None, obsRadius=None, S_goal_val=None, X_goal_val=None):
        if self.numObs > 0:
            if self.numObs != len(obsCentre) or self.numObs != len(obsRadius):
                print("Number of obstacles does not match the config file!", file=sys.stderr)
                sys.exit()
        if d_hat is None:
            d_hat = np.zeros(self.dyn.Nx).tolist()
        # ---- setting parameters ---- 
        p_ = []  # set to empty before reinitialize
        p_ += beta
        # print(f"{beta=}")
        p_ += x0
        p_ += self.X_nom_val[:, idx:(idx+self.TH)].elements()
        p_ += d_hat
        if self.useGoalFlag:
            if X_goal_val is None:
                p_ += x0
            else:
                p_ += X_goal_val
            # ----------------------
            if S_goal_val is None:
                p_ += [0.]*(self.TH-2)
                p_ += [1.]
            else:
                p_ += S_goal_val
        if self.linDyn:
            p_ += self.U_nom_val[:, idx:(idx+self.TH-1)].elements()
        if self.numObs > 0:
            for i_obs in range(self.numObs):
                p_.append(obsCentre[i_obs][0])
                p_.append(obsCentre[i_obs][1])
                p_.append(obsRadius[i_obs])
        # ---- Set warm start ----
        self.args.x0 = []
        if X_warmStart is not None:
            for i in range(self.TH-1):
                self.args.x0 += X_warmStart[:, i].elements()
                if U_warmStart is not None:
                    self.args.x0 += U_warmStart[:, i].elements()
                else:
                    self.args.x0 += [0.0]*self.dyn.Nu
            self.args.x0 += X_warmStart[:, -1].elements()
            # print(len(self.args.x0))
        else:
            for i in range(self.TH-1):
                self.args.x0 += self.X_nom_val[:, i].elements()
                if U_warmStart is not None:
                    self.args.x0 += U_warmStart[:, i].elements()
                else:
                    self.args.x0 += [0.0]*self.dyn.Nu
            self.args.x0 += self.X_nom_val[:, -1].elements()
        for i in range(self.Nphases):
            self.args.x0 += self.dyn.z0
        # ---- Solve the optimization ----
        start_time = time.time()
        # print('-----------------------------')
        # print(len(self.args.x0))
        # print(len(self.args.lbx))
        # print(len(self.args.ubx))
        # print(len(self.args.lbg))
        # print(len(self.args.ubg))
        # print(len(p_))
        # print('-----------------------------')
        sol = self.solver(
                x0=self.args.x0,
                lbx=self.args.lbx, ubx=self.args.ubx,
                lbg=self.args.lbg, ubg=self.args.ubg,
                p=p_)
        # print(sol)
        # sys.exit()
        # ---- save computation time ---- 
        t_opt = time.time() - start_time
        # ---- decode solution ----
        resultFlag = self.solver.stats()['success']
        opt_sol = sol['x']
        f_opt = sol['f']
        # get x_opt, u_opt, other_opt
        opt_sol_xu = opt_sol[:(self.TH*self.Nxu-self.dyn.Nu)]
        opt_sol_s = opt_sol[(self.TH*self.Nxu-self.dyn.Nu):]
        x_opt = []
        for i in range(self.dyn.Nx):
            x_opt = cs.vertcat(x_opt, opt_sol_xu[i::self.Nxu].T)
        u_opt = []
        for i in range(self.dyn.Nx, self.Nxu):
            u_opt = cs.vertcat(u_opt, opt_sol_xu[i::self.Nxu].T)
        other_opt = []
        if self.dyn.Nz > 0:
            for i in range(self.dyn.Nz):
                other_opt = cs.vertcat(other_opt, opt_sol_s[i::self.dyn.Nz].T)
        # ---- warm start ----
        # for i in range(0, self.dyn.Nx):
        #     opt_sol[i::self.Nopt] = [0.0]*(self.TH)
        for i in range(self.dyn.Nx, self.Nxu):
            opt_sol[:(self.TH*self.Nxu-self.dyn.Nu)][i::self.Nxu] = [0.]*(self.TH-1)
        opt_sol[(self.TH*self.Nxu-self.dyn.Nu):] = [0.]
        self.args.x0 = opt_sol.elements()
        # ---- add nominal trajectory ----
        if self.linDyn:
            x_opt += self.X_nom_val[:, idx:(idx+self.TH)]
            u_opt += self.U_nom_val[:, idx:(idx+self.TH-1)]

        return resultFlag, x_opt, u_opt, other_opt, f_opt, t_opt
