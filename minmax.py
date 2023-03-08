'''
optimization max of all force
interior point
'''

import math
import random
import numpy as np
from scipy.optimize import linprog
import time
'''
A = [sin(t1), sin(t2),...,sin(tn);
     cos(t1), cos(t2),...,cos(tn)]
B = [G;
     0]
x = [f1, f2,...,fn]

minimize sum(f1,f2,fn)
constraints
Ax=B
x>=0
'''

start = time.time()
thetas = [1/2,1/3,1/4,0]
num_cables = len(thetas)
F = 1
pi = math.pi
B = np.array([[F],[0]])
A = []
for i in thetas:
    A.append([math.sin(i*pi),math.cos(i*pi)])
A = np.transpose(A)
A_z = np.append(A,np.zeros((2,1)),axis=1)

Neg_con = np.append(np.identity(num_cables),-np.ones((num_cables,1)),axis=1)


C = np.append(np.zeros(num_cables),[1]) # n+1
start = time.time()
Res = linprog(c=C,A_ub= Neg_con,b_ub = np.zeros((num_cables,1)),A_eq=A_z,b_eq=B,bounds=(0,F),method = 'interior-point')
# print('runtime:',time.time()-start)
print('solution exist:',Res.success)
print('optimized F is ', Res.x[:-1])
print('optimized max F is', Res.x[-1])
