import math
import random
import numpy as np
from scipy.optimize import linprog
import time
import matplotlib.pyplot as plt
import pickle as pkl
pi = math.pi
def get_thetas(phir,y,phi1=pi/2,phi3=pi/4,r=1,l=2):
    h1 = (phi1+phir)*r
    h3 = (phi3+phir)*r
    if y==0:
        theta1 = theta3 = pi/2
        theta2 = math.atan(h1/(l-y))
        theta4 = math.atan(h3/(l-y))
    elif y==l:
        theta2=theta4=pi/2
        theta1 = pi- math.atan(h1/y)
        theta3 = pi- math.atan(h3/y)
    else:
        theta1 = pi- math.atan(h1/y)
        theta2 = math.atan(h1/(l-y))
        theta3 = pi- math.atan(h3/y)
        theta4 = math.atan(h3/(l-y))
    return [theta1,theta2,theta3,theta4]


def get_min_max_f(thetas,G_e):
    n=len(thetas)
    B = np.array([[G_e],[0]])
#     print(B)
    A = []
    for i in thetas:
        A.append([math.sin(i),math.cos(i)])
    A = np.transpose(A)
    A_z = np.append(A,np.zeros((2,1)),axis=1)
#     print(A_z)
    Neg_con = np.append(np.identity(n),-np.ones((n,1)),axis=1)
#     print(Neg_con)
    C = np.append(np.zeros(n),[1]) # n+1
#     print(C)
    start = time.time()
    Res = linprog(c=C,A_ub= Neg_con,b_ub = np.zeros((n,1)),A_eq=A_z,b_eq=B,bounds=(0,G_e),method = 'interior-point')
#     print(Res)
    return Res.x[-1]

resolution = 100
G = 100
l=2
r=1
phi1=pi/2
phi3=pi/4
pi = math.pi
phi_res = pi/2/(resolution-1)
y_res = l/(resolution-1)
f_map = np.zeros((resolution,resolution))
start = time.time()
for i in range(resolution):
    for j in range(resolution):
        phir = i*phi_res
        y= j*y_res
        G_e = G*math.sin(phir)
#         print(G_e)
        thetas = get_thetas(phir,y,r=r,l=l)
#         print(thetas)
        max_f = get_min_max_f(thetas,G_e)
#         print(max_f)
        f_map[i,j]=max_f
print('runtime',time.time()-start)
with open('p1p2p3p4.npy','wb') as f:
    np.save(f,f_map)
# pkl.dump(f_map,open( '{}_l_{}_r_{}_phi1_{}_phi3_{}.pl'.format(resolution,l,r,phi1/pi,phi3/pi), "wb" ))
plt.imshow(f_map, cmap="autumn_r",interpolation='nearest', vmin=0,vmax=G)
plt.colorbar()
plt.xlabel('l')
plt.ylabel('$\phi_r$')
plt.savefig('{}_l_{}_r_{}_phi1_{}_phi3_{}.png'.format(resolution,l,r,phi1/pi,phi3/pi))
plt.show()


'''
load map and print contour
'''
with open('p1p2p3p4.npy','rb') as f:
    l_map = np.load(f)
# f_map = pkl.load(open('100_l_2_r_1_phi1_0.5_phi3_0.25.pl','r'))
xlist = np.linspace(0, l, resolution)
ylist = np.linspace(0,pi/4, resolution)
X, Y = np.meshgrid(xlist, ylist)
CT = plt.contour(X,Y,l_map,vmin=0,vmax=G)
plt.clabel(CT, fmt = '%2.1d', colors = 'k', fontsize=14)
plt.gca().invert_yaxis()
plt.xlabel('l')
plt.ylabel('$\phi_r$')
plt.title('p1=($\pi/2$,0),p2=($\pi/2$,l),p3=($\pi/4$,0),p4=($\pi/4$,l)')
plt.show()
# plt.savefig('contour{}_l_{}_r_{}_phi1_{}_phi3_{}.png'.format(resolution,l,r,phi1/pi,phi3/pi))
