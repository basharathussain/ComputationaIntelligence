# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 18:04:30 2020

@author: Basharat
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/8/20
# @Author  : github.com/guofei9987

import numpy as np
from sko.tools import func_transformer
# Now Plot the animation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# PSO (Particle swarm optimization) algorithm
class PSO():
    """
    Parameters
    --------------------
    cost_func : function
        The func you want to do optimal
    dimension : int
        Number of dimension, which is number of parameters of func.
    N : int
        Size of population, which is the number of Particles. 
    max_iteration : int
        Max no of iterations

    Attributes
    ----------------------
    pbest_x : array_like, shape is (N,dim)
        best location of every particle in history
    pbest_y : array_like, shape is (N,1)
        best image of every particle in history
    gbest_x : array_like, shape is (1,dim)
        general best location for all particles in history
    gbest_y : float
        general best image  for all particles in history
    gbest_y_hist : list
        gbest_y of every iteration
    """

    def __init__(self, cost_func, dimension, population_size=40, max_iteration=150, lb=None, ub=None, w=0.8, c1=0.5, c2=0.5):
        self.func = func_transformer(cost_func)
        
        self.w = w                      # inertia weight
        self.c1, self.c2 = c1, c2       # parameters to control personal best, global best respectively
        self.N = population_size        # number of particles
        self.dimension = dimension      # dimension of particles, which is the number of variables of func
        self.max_iter = max_iteration   # max iter

        self.has_constraints = not (lb is None and ub is None)
        self.lb = -np.ones(self.dimension) if lb is None else np.array(lb)
        self.ub = np.ones(self.dimension) if ub is None else np.array(ub)
        v_high = self.ub - self.lb
      
        assert self.dimension == len(self.lb) == len(self.ub), 'dimension == len(lb) == len(ub) is not True'
        assert np.all(self.ub > self.lb), 'upper-bound must be greater than lower-bound'

        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.N, self.dimension)) # position of particles
        self.V = np.random.uniform(low=-v_high, high=v_high, size=(self.N, self.dimension))  # speed of particles
        self.Y = self.calculate_objfunc_Y()             # y = f(x) for all particles
        
        self.pbest_x = self.X.copy()                    # personal best location of every particle in history
        self.pbest_y = self.Y.copy()                    # best objective cost of every particle in history
        
        self.gbest_x = np.zeros((1, self.dimension))    # global best location for all particles
        self.gbest_y = np.inf                           # global best objective cost for all particles
        
        self.gbest_y_hist = []                          # gbest_y of every iteration
        self.update_gbest()

        # record verbose values
        self.record_mode = False
        self.record_value = {'X': [], 'V': [], 'Y': []}

    # Update the velocity to get new value
    def update_V(self):
        r1 = np.random.rand(self.N, self.dimension)
        r2 = np.random.rand(self.N, self.dimension)
        
        self.V = self.w * self.V + \
                 self.c1 * r1 * (self.pbest_x - self.X) + \
                 self.c2 * r2 * (self.gbest_x - self.X)
    
    # Update the position to get new value
    def update_X(self):
        self.X = self.X + self.V

        if self.has_constraints:
            self.X = np.clip(self.X, self.lb, self.ub)

    # Calculate objective function of all particles
    def calculate_objfunc_Y(self):
        # calculate y for every x in X
        self.Y = self.func(self.X).reshape(-1, 1)
        return self.Y

    # Local best position of all the particles
    def update_pbest(self):
        '''
        personal best
        :return:
        '''
        self.pbest_x = np.where(self.pbest_y > self.Y, self.X, self.pbest_x)
        self.pbest_y = np.where(self.pbest_y > self.Y, self.Y, self.pbest_y)
    
    # Global best position of all the particles
    def update_gbest(self):
        '''
        global best
        :return:
        '''
        if self.gbest_y > self.Y.min():
            self.gbest_x = self.X[self.Y.argmin(), :].copy()
            self.gbest_y = self.Y.min()

    # Track values by iteration to print later
    def recorder(self):
        if not self.record_mode:
            return
        self.record_value['X'].append(self.X)
        self.record_value['V'].append(self.V)
        self.record_value['Y'].append(self.Y)

    # Run PSO algorithm
    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        for iter_num in range(self.max_iter):
            self.update_V()
            self.recorder()
            self.update_X()
            self.calculate_objfunc_Y()
            self.update_pbest()
            self.update_gbest()

            self.gbest_y_hist.append(self.gbest_y)
        return self

    fit = run


#--- COST FUNCTIONS ------------------------------------------------------------+   
# function we are attempting to optimize (minimize)

def costFunc0(x):
    [x1, x2] = x
    return x1 ** 2 + (x2 - 0.05) ** 2

def costFunc1(x):
    total=0
    for i in range(len(x)):
        total+=x[i]**2
    return total

## https://www.researchgate.net/publication/261197555_A_generic_particle_swarm_optimization_Matlab_function
# function we are attempting to optimize (minimize)
def costFunc2(x):
    n = len(x)
    sum_1=0
    for i in range(len(x)):
        sum_1+=x[i]**2

    sum_2=0
    for i in range(len(x)):
        sum_2+= np.cos(2*np.pi*x[i]) 
   
    f = 20 + np.exp(1) - 20*np.exp(-0.2*np.sqrt((1/n)*sum_1)) - np.exp((1/n)*sum_2);
    return f

# https://docs.microsoft.com/en-us/archive/msdn-magazine/2013/september/test-run-multi-swarm-optimization
def costFunc3(x):
    f=0
    for i in range(len(x)):
        f += (x[i]**2) - (10 * np.cos(2 * np.pi * x[i]))
    return f + 20      


########################################
####### CHANGE THE FUNCTION BELOW ######
########################################
f = costFunc0

########################################
####### MAIN PSO call ######
########################################
#pso = PSO(func=demo_func, dimension=2, population_size=30, max_iter=100, lb=[-1, -1], ub=[1, 1])
pso = PSO(cost_func=f, dimension=2, population_size=30, max_iteration=100, lb=[-1, -1], ub=[1, 1])
#pso = PSO(cost_func=f, dimension=2, population_size=3, max_iteration=5, lb=[-1, -1], ub=[1, 1])
pso.record_mode = True
pso.run()
print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)




########################################
####### Plot PSO on 3D surfice ######
########################################
#==================================
# Plotting the pso
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_title('title', loc='center')
line = ax.plot([], [], 'b.')

X_grid, Y_grid = np.meshgrid(np.linspace(-1.0, 1.0, 40), np.linspace(-1.0, 1.0, 40))
Z_grid = f((X_grid, Y_grid))
ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='jet', linewidth=0, antialiased=True, alpha=0.5)
 
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)

plt.ion()
p = plt.show()

def animate(frame):
    i, j = frame // 10, frame % 10
    ax.set_title('iter = ' + str(i))
    X_tmp = X_list[i] + V_list[i] * j / 10.0
    plt.setp(line, 'xdata', X_tmp[:, 0], 'ydata', X_tmp[:, 1])
    return line

ani = FuncAnimation(fig, animate, blit=True, interval=40, frames=400)
plt.show() 



###########################################################
####### Plot PSO on contour and animate particles ######
###########################################################
record_value = pso.record_value
X_list, V_list = record_value['X'], record_value['V']

fig, ax = plt.subplots(1, 1)

# Plotting the pso
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

ax.set_title('title', loc='center')
line = ax.plot([], [], 'b.')

X_grid, Y_grid = np.meshgrid(np.linspace(-1.0, 1.0, 40), np.linspace(-1.0, 1.0, 40))
Z_grid = f((X_grid, Y_grid))
ax.contour(X_grid, Y_grid, Z_grid, 10)
 
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)

plt.ion()
p = plt.show()


def update_scatter(frame):
    i, j = frame // 10, frame % 10
    ax.set_title('iter = ' + str(i))
    X_tmp = X_list[i] + V_list[i] * j / 10.0
    plt.setp(line, 'xdata', X_tmp[:, 0], 'ydata', X_tmp[:, 1])
    return line


ani = FuncAnimation(fig, update_scatter, interval=25, frames=250, repeat=True)
plt.show()

#ani.save('pso.gif', writer='pillow')   

