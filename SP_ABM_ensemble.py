# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 09:11:18 2023

@author: sebzi
"""

import numpy as np

import matplotlib.pyplot as plt
from numba import jit, njit
import time

start = time.perf_counter()

#@njit(fastmath=True)
def ABM_ICs(N,XL,XU,OL,OU):
    
    X_pos = np.zeros(N,dtype=np.float64)
    
    O_pos =  np.zeros(N,dtype=np.float64)
    #generating the ICs of the individuals
    '''
    for i in range(N):
        X_pos[i] = np.random.normal(loc=0.0, scale = 0.5) #np.random.uniform(XL,XU)
        O_pos[i] = np.random.normal(loc=0.0, scale = 1.0) #np.random.uniform(OL,OU)

        #rng1 = set_seed(i)#np.random.default_rng(seed=i)
        #rng2 = set_seed(2*i)#np.random.default_rng(seed=2*i)

        #X_pos[i] = rng1.uniform(XL,XU)
        
       # O_pos[i] = rng2.uniform(OL,OU)
    ''' 
    
    means = [ [0.6, -0.95],[-0.92,0.81],[0.90,-0.63], [-0.53, 0.05]]
    var = [0.2, 0.25, 0.25, 0.15]
    
    init = 4
    
    for i in range(init):
        pos = np.random.multivariate_normal(means[i],[[var[i]**2,0],[0,var[i]**2]], size = int(N/init))
        
        X_pos[int(N/4)*i:int(N/4)*(i+1)] = pos[:,0]
        O_pos[int(N/4)*i:int(N/4)*(i+1)] = pos[:,1]
    
    
    #X_pos[int(N/2):] = np.random.normal(loc=-1, scale = 0.5,size = int(N/2))
    #O_pos[int(N/2):] = np.random.normal(loc=-1, scale = 0.5,size = int(N/2))    
    return X_pos, O_pos

@njit(fastmath=True)
def a(r,R):
    
    y = np.zeros(len(r),dtype=np.float64)
    for i in range(len(r)):
        
    
        if r[i]<=R:
            y[i]=1.0
        else:
            y[i]=0.0

    return y



@njit(fastmath=True)
def F(X,O,beta,alpha,R):
    N = len(X) 
    
    FU = np.zeros(N,dtype=np.float64)
    FV = np.zeros(N,dtype=np.float64)
    
    for i in range(N):
               
        diff_indi = X-X[i]
        
        r = np.abs(diff_indi)
        
        nearby = a(r,R)
        
        store = nearby *np.sign(O[i]*O[:])
        
        FU[i] = beta*np.sum(store*diff_indi)/N 
        
        FV[i] = alpha*np.sum(nearby*(O-O[i]))/N 
    
    return FU, FV

@njit(fastmath=True)
def sigma(X,O,R):
    return np.sqrt(0.02)

@njit(fastmath=True)
def updateABM(Xi,Oi,dt,beta,alpha,R):
    N = len(Xi) 
    FX, FO = F(Xi,Oi,beta,alpha,R)
    
    Xf = np.zeros(N,dtype=np.float64)
    Of = np.zeros(N,dtype=np.float64)
    #Euler Maruyama scheme for individuals  
    Xf = Xi + FX*dt + sigma(Xi,Oi,R)*np.sqrt(dt)*np.random.standard_normal(N)
    
    Of = Oi + FO*dt + sigma(Xi,Oi,R)*np.sqrt(dt)*np.random.standard_normal(N)

    return Xf, Of


#@njit(fastmath=True)
def run(N,xL,xU,oL,oU,t,dt):
    
    Xi, Oi = ABM_ICs(N, xL,xU,oL,oU)
       
    time = np.arange(0,t,dt)
    
    X_t = np.zeros((len(time)+1, N)) #arrays for storing information at each time step
    O_t = np.zeros((len(time)+1, N))
    
    X_t[0]= Xi
    O_t[0] = Oi

    beta = 5
    alpha = 5
    R = 0.15

    
    for count in range(len(time)):
                           
        Xi, Oi = updateABM(Xi,Oi,dt,beta,alpha,R)
        
           
        X_t[count+1] = Xi
        O_t[count+1] = Oi


    return X_t, O_t

#@njit(fastmath=True)
def ensemble(T,save_t,I,N):
    
    
    storeX = np.zeros( (I,len(save_t),N),dtype=np.float64)
    storeO = np.zeros( (I,len(save_t),N),dtype=np.float64)    

    for i in range(I):
        solX, solO = run(N,-2.0,2.0,-2.0,2.0,T,0.01)
        
        for t in range(len(save_t)):
            solXt , solOt = solX[int(save_t[t]*100)] , solO[int(save_t[t]*100)]
            
            storeX[i,t] = solXt
            storeO[i,t] = solOt            
    
    return storeX , storeO


#time = np.arange(0,1.0+0.01,0.01)
saveT = [0,1,2]
X, O = ensemble(2,saveT,100,1000)

end = time.perf_counter()
print(end - start)

#PlotTrajec(solO, time)

#PlotTrajec(solX, time)

from sklearn.neighbors import KernelDensity
def kde2D(x, y, bandwidth, xbins=81j, ybins=81j, **kwargs): 
    """Build 2D kernel density estimate (KDE)."""

    # create grid of sample locations (default: 100x100)
    xx, yy = np.mgrid[-2:2:xbins, 
                      -2:2:ybins]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train  = np.vstack([y, x]).T

    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde_skl.score_samples(xy_sample))
    return xx, yy, np.reshape(z, xx.shape)


bandwidth = [0.1,0.03,0.015]

fig, axs = plt.subplots(1, 3,figsize=(18,4))
csfont = {'fontname':'DejaVu Sans'}
axs[0].set_ylabel("ABM",**csfont)
for j in range(len(saveT)):
    
    Xt = X[:,j,:]
    Ot = O[:,j,:]
    
    z_m = np.zeros((81,81))

    for i in range(100):
    
        xx, yy, zz = kde2D(Xt[i],Ot[i], bandwidth[j])
           
        z_m += zz/100

    #plt.figure(figsize=(12,8))
    pcm = axs[j].pcolormesh(xx, yy, z_m,cmap='inferno')
    axs[j].set_yticks([-2,-1,0,1,2])
    axs[j].set_yticklabels([-2,-1,0,1,2],fontname = 'DejaVu Sans')
    axs[j].set_xticks([-2,-1,0,1,2])
    axs[j].set_xticklabels([-2,-1,0,1,2],fontname = 'DejaVu Sans')
    #axs[j].colorbar(pad=0.02)
    fig.colorbar(pcm,ax=axs[j])
   # clb.ax.tick_params(fontname = 'DejaVu Sans')
    #plt.tight_layout()
    #plt.savefig('D:\\New_folder\\PhD\\test' + str(j) +'.png', dpi=300, bbox_inches='tight')
fig.tight_layout()    

'''
plt.figure(figsize=(12,8))
plt.pcolormesh(xx, yy, z_m,cmap='inferno')
plt.yticks([-2,-1,0,1,2])
plt.xticks([-2,-1,0,1,2])
plt.colorbar(pad=0.02)
plt.tight_layout()
#plt.savefig('D:\\New_folder\\PhD\\test.png', dpi=300, bbox_inches='tight')

fig, axes = plt.subplots(figsize=(12,8))
sc = axes.pcolormesh(xx, yy, z_m,cmap='inferno')
#fig.yticks([-2,-1,0,1,2])
#fig.xticks([-2,-1,0,1,2])
fig.colorbar(sc, ax=axes, pad=0.01)
fig.tight_layout()

from sklearn.model_selection import GridSearchCV

bandwidth = np.arange(0.05, 0.5, .05)
kde = KernelDensity(kernel='gaussian')
grid = GridSearchCV(kde, {'bandwidth': bandwidth})
grid.fit(Xt[0],Ot[0])

kde = grid.best_estimator_
print("optimal bandwidth: " + "{:.2f}".format(kde.bandwidth))


Xi = X[:,0,:]
Xf = X[:,1,:]
Oi = O[:,0,:]
Of = O[:,1,:]

z_mi = np.zeros((81,81))
z_mf = np.zeros((81,81))

for i in range(100):

    xx, yy, zz = kde2D(Xi[i],Oi[i], 0.1)

    xx, yy, zzf = kde2D(Xf[i],Of[i], 0.03)
    
    z_mi += zz/100
    z_mf += zzf/100


plt.figure()
plt.pcolormesh(xx, yy, z_mi,cmap='inferno')
plt.colorbar()

plt.figure()
plt.pcolormesh(xx, yy, z_mf,cmap='inferno')
plt.colorbar()    
'''

'''   
Xi_tot = np.sum(Xi,axis=0)/100   
Oi_tot = np.sum(Oi,axis=0)/100  

plt.figure()
plt.scatter(Xi_tot,Oi_tot)
plt.xlim(-2,2)
plt.ylim(-2,2)

xx, yy, zz_it = kde2D(Xi_tot,Oi_tot, 0.2)
plt.figure()
plt.pcolormesh(xx, yy, zz_it,cmap='inferno')
plt.colorbar()


Xf_tot = np.sum(Xf,axis=0)/100   
Of_tot = np.sum(Of,axis=0)/100  

plt.figure()
plt.scatter(Xf_tot,Of_tot)
plt.xlim(-2,2)
plt.ylim(-2,2)

xx, yy, zz_ft = kde2D(Xf_tot,Of_tot, 0.2)
plt.figure()
plt.pcolormesh(xx, yy, zz_ft,cmap='inferno')
plt.colorbar()
'''