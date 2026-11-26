"""IPOPT reference solver of the RPINN paper (GEKKO, MPCC Weymouth).

Ported verbatim from `RPINN/Notebooks/original/Classic_solution.ipynb`; the only
addition is `max_time`, a per-solve wall-time cap in seconds.
"""
import numpy as np
import pandas as pd
from gekko import GEKKO
from scipy.sparse import coo_matrix, eye, hstack
from sklearn.metrics import mean_squared_error

mse = mean_squared_error


class Natural_gas_flow:

    def __init__(self, data, disp=False, weymouth_type='MPCC',
                 num_approximations=20,initial=None, max_time=None):
        """
        In the initialization,  the xlsx file that contains the network data is
        loaded, the information is extracted from this same file, the incidence
        matrix is built, which describes the topology of the system and finally the
        upper and lower limits are established in each element of the system.
        """
        self.data = data
        self.max_time = max_time
        self.disp = disp
        self.num_approximations = num_approximations
        self.weymouth_type = weymouth_type
        # The information is extracted from the xlsx file
        self.node_info = pd.read_excel(self.data, sheet_name='node.info')
        self.node_info['node_id'] = self.node_info['node_id'].astype('int')
        self.node_info['type'] = self.node_info['type'].astype('int')

        self.node_dem = pd.read_excel(self.data, sheet_name='node.dem')
        self.node_dem['Total'] = self.node_dem.sum(axis=1)
        self.node_user = self.node_dem[self.node_dem['Total'] != 0]
        self.loads = self.node_dem['Total'].values
        self.node_demcost = pd.read_excel(self.data, sheet_name='node.demcost')

        self.well = pd.read_excel(self.data, sheet_name='well')
        self.well['node'] = self.well['node'].astype('int')

        self.node_deminitial = pd.read_excel(self.data, sheet_name='node.deminitial')

        self.pipe = pd.read_excel(self.data, sheet_name='pipe')
        self.pipe['fnode'] = self.pipe['fnode'].astype('int')
        self.pipe['tnode'] = self.pipe['tnode'].astype('int')

        self.comp = pd.read_excel(self.data, sheet_name='comp')
        self.comp['fnode'] = self.comp['fnode'].astype('int')
        self.comp['tnode'] = self.comp['tnode'].astype('int')
        self.comp['Type'] = self.comp['Type'].astype('int')

        self.sto = pd.read_excel(self.data, sheet_name='sto')
        self.sto['node'] = self.sto['node'].astype('int')
        try:
            self.coordinates = pd.read_excel(self.data, sheet_name='coordinates')
        except:
            self.coordinates = False
        self.max_ratio = self.comp['ratio']
        self.N = len(self.node_info)
        self.W = len(self.well)
        self.P = len(self.pipe)
        self.C = len(self.comp)
        self.UNS = len(self.node_demcost) * len(self.node_demcost.T)
        self.S = len(self.sto)
        self.Kij = self.pipe['Kij'].values
        N = len(self.node_info)
        # Incidence Matrix is created
        # Wells ok
        df_wells = pd.read_excel(self.data, sheet_name='well')
        W = len(self.well)
        wells = coo_matrix(
            (np.ones(W, ), (self.well['node'] - 1, np.arange(W))), shape=(N, W))
        # Pipes ok
        df_pipes = pd.read_excel(self.data, sheet_name='pipe')

        P = len(self.pipe)
        data = np.concatenate((-1.0 * np.ones(P), np.ones(P)))
        row = pd.concat((self.pipe['fnode'] - 1, self.pipe['tnode'] - 1))
        col = np.concatenate(2 * [np.arange(P)])

        pipes = hstack(
            2 * [coo_matrix((data, (row, col)), shape=(N, P))]).toarray()
        # print('Pipes:', pipes.shape)
        # Compressors ok
        C = len(self.comp)
        data = np.concatenate((-1.0 * np.ones(C), np.ones(C)))
        row = pd.concat((self.comp['fnode'] - 1, self.comp['tnode'] - 1))
        col = np.concatenate(2 * [np.arange(C)])
        comps = coo_matrix((data, (row, col)), shape=(N, C))
        # Users ok
        users = hstack(len(self.node_demcost.T) * [eye(N)])
        # Storage
        self.sto = pd.read_excel(self.data, sheet_name='sto')
        S = len(self.sto)
        sto = coo_matrix(
            (np.ones(S, ), (self.sto['node'] - 1, np.arange(S))), shape=(N, S))
        sto = hstack([sto, -1.0 * sto])

        self.Minc = (hstack((wells, pipes, comps, users, sto))).toarray()
        # The limits of the elementes of the network are stablished
        fp = self.pipe['FG_O'] * (self.pipe['FG_O'] > 0)
        fn = self.pipe['FG_O'] * (self.pipe['FG_O'] < 0)
        f_ext = np.concatenate((fp, fn))

        # print(np.array(self.node_deminitial).flatten(order='F').shape)
        self.lb = np.concatenate((self.well['Imin'].values,
                                  [0] * len(self.pipe),
                                  self.pipe['Fg_min'].values,
                                  [0] * len(self.comp),
                                  [0] * len(self.node_dem) *
                                  (len(self.node_dem.T) - 1),
                                  [0] * len(self.sto) * 2,
                                  self.node_info['Pmin'].values))
        if np.any(initial==None):
            self.initial = np.concatenate((self.well['I'].values,
                                       f_ext,
                                       self.comp['fgc'].values,
                                       np.array(self.node_deminitial).flatten(order='F'),
                                       [0] * len(self.sto.values),
                                       [0] * len(self.sto.values),
                                       self.node_info['p'].values))
        else:
            try:
                if initial.shape[0]==self.lb.shape[0]:
                    self.initial=initial
                else:
                    self.initial = np.concatenate((self.well['I'].values,
                                       f_ext,
                                       self.comp['fgc'].values,
                                       np.array(self.node_deminitial).flatten(order='F'),
                                       [0] * len(self.sto.values),
                                       [0] * len(self.sto.values),
                                       self.node_info['p'].values))
            except:
                self.initial = np.concatenate((self.well['I'].values,
                                       f_ext,
                                       self.comp['fgc'].values,
                                       np.array(self.node_deminitial).flatten(order='F'),
                                       [0] * len(self.sto.values),
                                       [0] * len(self.sto.values),
                                       self.node_info['p'].values))



        self.ub = np.concatenate((self.well['Imax'].values,
                                  self.pipe['Fg_max'].values,
                                  [0] * len(self.pipe),
                                  self.comp['fmaxc'],
                                  self.node_dem['Res'],
                                  self.node_dem['Ind'],
                                  self.node_dem['Com'],
                                  self.node_dem['NGV'],
                                  self.node_dem['Ref'],
                                  self.node_dem['Pet'],
                                  self.sto['V0'] - self.sto['Vmax'].values,
                                  self.sto['Vmax'] - self.sto['V0'].values,
                                  self.node_info['Pmax'].values))

        self.objective_function()

    def objective_function(self):
        # Initialize Model
        self.m = GEKKO(remote=False)
        self.m.options.SOLVER =3 # IPOPT Solver

        cost = np.concatenate((self.well['Cg'].values,
                               self.pipe['C_O'].values,
                               -1 * self.pipe['C_O'].values,
                                self.comp['costc'].values,
                               self.node_demcost['al_Res'].values,
                               self.node_demcost['al_Ind'].values,
                               self.node_demcost['al_Com'].values,
                               self.node_demcost['al_NGV'].values,
                               self.node_demcost['al_Ref'].values,
                               self.node_demcost['al_Pet'].values,
                               (self.sto['C_S+'] - self.sto['C_V']).values,
                               -1 * (self.sto['C_S-'] -
                                     self.sto['C_V']).values,
                               self.sto['C_V'].values))

        NV = (len(self.well) + 2 * len(self.pipe) + len(self.comp) +
              (len(self.node_demcost) * len(self.node_demcost.T)) +
              2 * len(self.sto) + len(self.node_info))

        self.NV_obj = NV - len(self.node_info)

        limits = (self.initial, self.lb, self.ub)

        x = [self.m.Var(value=value,lb=lb,ub=ub) for value, lb, ub in zip(self.initial, self.lb, self.ub)]
        # x = [self.m.Var(lb=lb, ub=ub) for lb, ub in zip(self.lb, self.ub)]
        self.X = np.array(x)


        self.f_plus = [self.m.Var(lb=lb, ub=ub) for lb, ub in zip([0]*len(self.pipe), self.pipe['Fg_max'].values)]
        self.f_minus = [self.m.Var(lb=lb, ub=ub) for lb, ub in zip([0]*len(self.pipe), self.pipe['Fg_max'].values)]
        self.y = [self.m.Var() for i in self.pipe['Fg_max']]
        # for i, value in enumerate(np.array(self.initial)):
        #     self.m.fix_initial(x[i], value)

        # self.X = np.array(x)

        # B_lb = np.array([0] * 2 * len(self.pipe))
        # B_ub = np.array([1] * 2 * len(self.pipe))

        # B = [self.m.Var(lb=lb, ub=ub, integer=True)
        #      for lb, ub in zip(B_lb, B_ub)]
        # self.B = np.array(B)

        # for x, x0 in zip(self.X, self.initial):
        #   self.m.fix_initial(x, x0)

        self.J = cost.T @ np.concatenate((self.X[:self.NV_obj],
                                          self.sto['V0'].values))

        self.constraints()
        self.solve_network()

    def constraints(self, ):
        self.gas_balance()
        self.comp_ratio()

        if self.weymouth_type == 'MPCC':
            self.weymouth_MPCC()
        elif self.weymouth_type == 'SOC':
            self.weymouth_SOC()
        elif self.weymouth_type == 'Taylor':
            self.weymouth_Taylor()
        elif self.weymouth_type == 'tanh':
            self.weymouth_tanh()
    # (2.24)
    def gas_balance(self):
        ftrans_max = self.pipe['Fg_max'].values
        pipe_indx1 = len(self.well)
        pipe_indx2 = len(self.well) + len(self.pipe)

        # self.m.Equations([self.X[pipe_indx1+i] + self.X[pipe_indx2+i] <= j for i, j in enumerate(ftrans_max)])
        #self.m.Equations([self.net_flow(self.X)[i] <=
        #                 j for i, j in enumerate(ftrans_max)])

        self.A = self.X[:self.NV_obj] @ self.Minc.T
        node=int(len(self.node_dem))
        for i in range(node):
            load = self.m.Param(value=self.loads[i])
            self.m.Equation(self.A[i] == load)

    def gas_balance_eval(self):
        sol = self.X[:self.NV_obj]
        self.x_sol = np.array([val.value[0] for val in sol])
        self.eval = self.x_sol @ self.Minc.T - self.loads
        return self.eval

    def comp_ratio(self):
        press = self.X[-self.N:]
        subA = self.Minc[:, self.W + 2 * self.P:self.W + 2 * self.P + self.C]
        g = np.prod(press.reshape(-1, 1) ** subA, 0)
        g2 = self.m.Param(value=1)

        for i, constrain in enumerate(g):
            g1 = self.m.Param(value=self.max_ratio[i])
            self.m.Equation(constrain <= g1)
            self.m.Equation(constrain >= g2)

    def net_flow(self, x):
        flow = x[self.W:self.W + self.P] + \
            x[self.W + self.P:self.W + 2 * self.P]
        return flow

    def weymouth_MPCC(self):
        f = (self.net_flow(self.X).reshape(-1,))
        press = np.array(self.X[-self.N:])
        press2 = np.array(self.X[-self.N:]) ** 2

        for i in range(self.P):

            i_index = (self.pipe['fnode'] - 1)[i]
            j_index = (self.pipe['tnode'] - 1)[i]
            p = press2[i_index] - press2[j_index]
            K = self.Kij[i]#f[i]== self.m.sign2(p) * self.m.sqrt( K* self.m.abs2(p))
            self.m.Equation(f[i]== self.m.sign2(p) * self.m.sqrt( K* self.m.abs2(p)))#self.m.sign2(f[i]) * f[i]**2 == p*K)

    def weymouth_tanh(self):
        f = (self.net_flow(self.X).reshape(-1,))
        press = np.array(self.X[-self.N:])
        press2 = np.array(self.X[-self.N:]) ** 2

        for i in range(self.P):

            i_index = (self.pipe['fnode'] - 1)[i]
            j_index = (self.pipe['tnode'] - 1)[i]
            p = press2[i_index] - press2[j_index]
            K = self.Kij[i]
            self.m.Equation(self.m.tanh(f[i]) * f[i]**2 == p*K)

    def weymouth_SOC(self):
        print('SOC')
        f = self.net_flow(self.X).reshape(-1,)
        press = self.X[-self.N:]
        Minc_P = self.Minc[:, self.W:self.W+self.P]
        Minc_P_i = Minc_P.copy()
        Minc_P_j = Minc_P.copy()
        Minc_P_i[Minc_P_i > 0] = 0
        Minc_P_i = -1*Minc_P_i
        Minc_P_j[Minc_P_j < 0] = 0
        p_min = self.node_info['Pmin']
        p_max = self.node_info['Pmax']
        p_min_i = p_min @ (Minc_P_i)
        p_max_j = p_max @ (Minc_P_j)
        p_max_i = p_max @ (Minc_P_i)
        p_min_j = p_min @ (Minc_P_j)
        phi_plus_lowlimit = p_min @ np.abs(Minc_P)
        phi_plus_uplimit = p_max @ np.abs(Minc_P)
        phi_minus_lowlimit = p_min_i - p_max_j
        phi_minus_uplimit = p_max_i - p_min_j

        phi_plus = (press @ np.abs(Minc_P))
        phi_minus = (press @ (-1*Minc_P))


        y = [self.m.Var(integer=True) for i in range(self.P)]
        Phi = [self.m.Var() for i in range(self.P)]
        M = [self.m.Var(lb=1) for i in range(self.P)]


        Phi_lowlimit1 = (phi_plus_lowlimit * phi_minus) + \
                    (phi_plus * phi_minus_lowlimit) - \
                    (phi_plus_lowlimit * phi_minus_lowlimit)

        Phi_lowlimit2 = (phi_plus_uplimit * phi_minus) + \
                    (phi_minus * phi_minus_uplimit) - \
                    (phi_plus_uplimit * phi_minus_uplimit)


        Phi_uplimit1 = (phi_plus_uplimit * phi_minus) + \
                    (phi_plus * phi_minus_lowlimit) - \
                    (phi_plus_uplimit * phi_minus_lowlimit)

        Phi_uplimit2 = (phi_plus_lowlimit * phi_minus) + \
                    (phi_plus * phi_minus_uplimit) - \
                    (phi_plus_lowlimit * phi_minus_uplimit)

        for i in range(self.P):
            self.m.Equation(phi_plus[i] <= self.m.Param(value = phi_plus_uplimit[i]))
            self.m.Equation(phi_plus[i] >= self.m.Param(value = phi_plus_lowlimit[i]))
            self.m.Equation(phi_minus[i] <= self.m.Param(value = phi_minus_uplimit[i]))
            self.m.Equation(phi_minus[i] >= self.m.Param(value = phi_minus_lowlimit[i]))

            self.m.Equation(Phi[i] <= self.m.Param(value = Phi_uplimit1[i]))
            self.m.Equation(Phi[i] <= self.m.Param(value = Phi_uplimit2[i]))
            self.m.Equation(Phi[i] >= self.m.Param(value = Phi_lowlimit1[i]))
            self.m.Equation(Phi[i] >= self.m.Param(value = Phi_lowlimit2[i]))

            self.m.Equation(self.m.Param(value = f[i]) <= self.Kij[i] * Phi[i] + \
                            M[i]**2 * (1 - y[i]))

            self.m.Equation(self.m.Param(value = f[i]) <= -1*self.Kij[i] * Phi[i] + \
                            M[i]**2 * y[i])

    def weymouth_Taylor(self):
        print('Taylor')
        num_pipes = len(self.pipe)
        num_approximations = self.num_approximations
        press = self.X[-self.N:]
        f = self.net_flow(self.X).reshape(-1,)

        for i in range(num_pipes):
            low_limit = self.node_info['Pmin'].values[i]
            upp_limit = self.node_info['Pmax'].values[i]
            K = self.Kij[i]
            pi = press[self.pipe['fnode'][i] - 1]
            pj = press[self.pipe['tnode'][i] - 1]
            g_value = self.m.Param(value = f[i])

        for j in range(num_approximations):
            PO, PI = np.sort(np.random.uniform(low=low_limit, high=upp_limit, size=(2,)))
            g = K*( (PI/np.sqrt(PI**2-PO**2))*pi + (PO/np.sqrt(PI**2-PO**2))*pj)
            self.m.Equation(self.m.abs(g_value) <= g)



    def weymouth_eval(self):
        sol = self.X[:self.NV_obj]
        self.x_sol = np.array([val.value[0] for val in sol])
        f_plus = self.x_sol[self.W: self.W+self.P]
        f_minus = self.x_sol[self.W+self.P : self.W+self.P+self.P]

        f_net = f_plus + f_minus
        p2 = np.array([ (p[0])**2 for p in self.X[-self.N:]])
        self.pipe_f_node = self.pipe['fnode'] - 1
        self.pipe_t_node = self.pipe['tnode'] - 1
        K = self.pipe['Kij'].values
        p2_sub = p2[self.pipe_f_node] - p2[self.pipe_t_node]
        self.wey_eval = (K * p2_sub) - np.sign(f_net) * f_net**2

        return self.wey_eval

    def solve_network(self):
        self.m.options.OTOL = 1e-7
        self.m.options.MAX_ITER = 1e9
        if self.max_time is not None:
            self.m.options.MAX_TIME = self.max_time  # cap per sample (not in the paper)
        self.m.solver_options = ['mu_strategy adaptive',
                                 'constr_viol_tol 1e-7',
                                 'acceptable_tol 1e-7', ]
        #  'bound_push 1e-10',\
        #  'bound_frac 1e-10']
        self.m.Minimize(self.J)
        #self.m.open_folder()
        a=self.m.solve(disp=self.disp)
    def get_values(self):
        N = self.W + self.P + self.C + self.UNS + self.N
        names = [0] * N
        values = [0] * N

        for i in range(self.W):
            inj = self.well['node'].values[i]
            names[i] = 'Well ' + str(inj)
            values[i] = self.X[i].value[0]

        f_plus = self.X[self.W: self.W + self.P]
        f_minus = self.X[self.W + self.P: self.W + self.P + self.P]
        for i in range(self.P):
            ind = self.W + i
            From = self.pipe['fnode'][i]
            To = self.pipe['tnode'][i]
            names[ind] = 'Net flow pipe ' + str(From) + '-' + str(To)
            values[ind] = f_plus[i][0] + f_minus[i][0]

        for i in range(self.C):
            ind = self.W + self.P + i
            From = self.comp['fnode'][i]
            To = self.comp['tnode'][i]
            names[ind] = 'Flow compressor ' + str(From) + '-' + str(To)
            values[ind] = self.X[self.W + 2 * self.P + i].value[0]

        k = 0
        for i in range( len(self.node_demcost.T)):
            for j in range(len(self.node_demcost)):
                ind = self.W + self.P + self.C + k
                names[ind] = 'Shortage in node ' + str(j+1) + '-' + 'Load ' + str(i+1)
                values[ind] = self.X[self.W + 2 * self.P + self.C+ k].value[0]
                k += 1

        for i in range(self.N):
            ind = self.W + self.P + self.C + self.UNS + i
            names[ind] = 'Press in node  ' + str(i+1)
            values[ind] = self.X[-self.N+i].value[0]
        names.insert(0, 'Objective')
        values.insert(0, self.m.options.objfcnval)
        self.values=values
        return [self.m.options.SOLVETIME,self.values]
    def Evaluate(self):
        y_pred=self.weymouth_eval()
        W=mse(np.zeros(y_pred.shape),y_pred)
        y_pred=self.gas_balance_eval()
        B=mse(np.zeros(y_pred.shape),y_pred)
        S=self.X[-8:]
        #return  self.max_ratio
        ratio=np.zeros((2))
        ratio[0]=np.maximum(0,S[2][0]-self.max_ratio[0]*S[1][0])
        ratio[0]=np.maximum(0,S[3][0]-self.max_ratio[0]*S[2][0])
        ratio=mse(np.zeros(ratio.shape),ratio)
        return [B,W,ratio,self.get_values()]
