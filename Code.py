Created on Fri Jul 19 10:21:39 2024

@author: malhotran
"""
import numpy as np
from matplotlib import pyplot as plt
#from numba import njit
from scipy.stats import pareto
from scipy import stats
#import math
from numpy import linalg as LA
#import pandas as pd
#import seaborn as sb
#import random

import statsmodels.api as sm
from sklearn.linear_model import LinearRegression




# Distance on the torus




def distancematrix(M):
   distancematrix = np.empty(shape=(M, M), dtype=float)
   for i in range(M):
       distancematrix[i, i] = 0
       for j in range(i+1,M):
            distancematrix[i, j] = distancematrix[j,i]=min(j-i,M-j+i)
   return distancematrix



#%%
# Construction of the adjacency matrix

"Construction of the attachment probabilities"
def proba_matrix(alpha,t,N,sig):
    dT = distancematrix(N)
    #Uncomment below for same W
    #np.random.seed(10)
    W = pareto.rvs(t, size=N) #sampling Paretos with P(W>x)=1/x**{t} so t=tau-1 in our language
    proba_matrix =  np.empty(shape=(N, N), dtype=float)
    for i in range(N):
        proba_matrix[i,i]=0
        for j in range(i+1,N):
             proba_matrix[i, j]=proba_matrix[j,i] = min(max(W[i],W[j])*min(W[i],W[j])**sig/dT[i,j]**alpha,1)
    return proba_matrix

"Construction of adjacency matrix"

def adj_matrix(alpha,t,N,sig):
    adj_matrix = np.empty(shape=(N, N), dtype=float)
    pro_matrix = proba_matrix(alpha,t,N,sig)
    for i in range(N):
        adj_matrix[i,i]=0
        for j in range(i+1,N):
            adj_matrix[i, j]=adj_matrix[j, i]=np.random.binomial(1,pro_matrix[i, j]) ## I compute the Bernoulli's in each entry as Binomial(1,p)
    return adj_matrix
#%% Calculating eigenvalues of the centered matrix

def adj_mat(P):
    adj_matrix = np.empty(shape=(len(P), len(P)), dtype=float)
    for i in range(len(P)):
        adj_matrix[i,i]=0
        for j in range(i+1,len(P)):
            adj_matrix[i, j]=adj_matrix[j, i]=np.random.binomial(1,P[i, j]) ## I compute the Bernoulli's in each entry as Binomial(1,p)
    return adj_matrix
    
def eigenvalues_SFP_cent(alpha,t,N,sig,n_matrices):
    eig = np.zeros([n_matrices, N])
    summa = np.empty(shape=(n_matrices, ), dtype=float)
    for i in range(n_matrices):
        P = proba_matrix(alpha,t,N,sig)
        eig[i, :] = LA.eigvalsh(adj_mat(P)-P)
        summa[i] = np.sum(eig[i, :]**2)/N #Tr(A^2)=1/N sum lambda_i^2
    normalizing=np.sqrt(np.mean(summa))
    eig=eig/normalizing
    return eig, summa

#%%%%% Taking DAD for A a GUE matrix, D=sqrt(Pareto)

#%% D
def Diag(t,N):
    D=np.diag(pareto.rvs(t, size=N)**0.5)
    return D
#%% defining GUE matrix
def gue_eigs(N):
    # N: size of the matrix.    
    A = np.random.randn(N, N) + 1j * np.random.randn(N, N) 
    B = 2**-1 * (A + np.conj(A.T))/np.sqrt(N)
    D = np.real(np.linalg.eigvals(B))
    return D, B

def DAD(t,N,n_matrices):
    eigens_normal= np.zeros([n_matrices,N])
    summa = np.empty(shape=(n_matrices, ), dtype=float)
    for i in range(n_matrices):
        D=Diag(t,N)
        matr=np.matmul(np.matmul(D,gue_eigs(N)[1]),D)
        eigens=np.real(np.linalg.eigvals(matr))
        summa[i] = np.sum(eigens**2)/N #Tr(A^2)=1/N sum lambda_i^2
    normalizing=np.sqrt(np.mean(summa))
    eigens_normal=eigens/normalizing
    return eigens_normal, summa
#%%Calculating normalization for DAD and SFP (currently not used later in the notebook)
def const_DAD(t,N,n_matrices):
    empirical_trace = np.empty(shape=(n_matrices, ), dtype=float)
    for i in range(n_matrices):
        empirical_trace[i] = np.sum(DAD(t,N,n_matrices)[0]**2)/N #compute tr(A^2)/N 
    const = np.sqrt(np.sum(empirical_trace)/n_matrices) #computes the mean of the traces 
    return const 

def const_SFP(alpha,t,N,sig,n_matrices):
    empirical_trace = np.zeros(shape=(n_matrices, ), dtype=float)
    for i in range(n_matrices):
        empirical_trace[i] = np.sum((eigenvalues_SFP_cent(alpha,t,N,sig,n_matrices)[0])**2)/N #compute tr(A^2)/N 
    const = np.sqrt(np.sum(empirical_trace)/n_matrices) #computes the mean of the traces 
    return const,empirical_trace

#%%% Plotting the eigenvalues of DAD and SFP
def plot_random_matrix_eigenvalues(alpha, t , N, sig , n_matrices):
    eig_D, summaDAD = DAD(t,N,n_matrices)
    eig_SFP, summaSFP =eigenvalues_SFP_cent(alpha,t,N,sig,n_matrices)
    plt.figure(figsize = (10, 5))
    plt.title("ESD")
    plt.hist(eig_D.ravel(), bins=80, ec='black',density = True,color='b',alpha=0.5)
    plt.hist(eig_SFP.ravel(), bins=80, ec='black',density = True,color='g',alpha=0.5)
    plt.grid(True)
    #print(summaDAD,sumplomaSFP)
    plt.legend(["P*G*P", "KBRG"])
    plt.title('size='+str(N)+ ', alpha=' +str(alpha)+ ', tau='+ str(t+1)+ ', sigma='+str(sig) )
    plt.show()
    
#%%%%% T'Setting the parameters'
alphav=0.1
tv=4
Nv=6000
sigv=0.2
n_matricesv=1
#%%'Calculating sf'


x = np.sort(eigenvalues_SFP_cent(alphav, tv, Nv, sigv, n_matricesv)[0].ravel())
y = 1-np.arange(1,len(x)+1)/float(len(x))

#%%x vector
truncateup=1.5
truncatedown=2.5
xs=np.log(x[(truncateup<x) & (x<truncatedown)])
fs=-2*tv*xs[:-1]
#%%% find the truncation for y
m=min(i for i in x if i > truncateup) #finds the first eigenvalue >truncateup (otherwise log is infty)
teta=np.where(x==m) #finds the index for eigenvalue above
#%%plot of log x and log P(X>x)
plt.plot(xs[:-1], np.log(y[teta[0][0]:-1]),"-b",label="log μ(x,∞)")
plt.plot(xs[:-1],fs,"-r",label="y=-2(τ-1)x")
plt.legend(loc="upper right")
plt.title( 'size=' +str(Nv)+ ', alpha=' +str(alphav)+ ', tau='+str(tv+1)+', sigma='+str(sigv) )
#plt.xlim([1, 2])
#plt.ylim([0, 0.2])
plt.show()
#%%% Simulating Gaussians (alpha=0)

#%%% Matrix of the r_{xy}
def proba_matrix1(alpha,t,N,sig):
    dT = distancematrix(N)
    #Uncomment below for same W
    #np.random.seed(10)
    W = pareto.rvs(t, size=N) #sampling Paretos with P(W>x)=1/x**{t} so t=tau-1 in our language
    proba_matrix =  np.empty(shape=(N, N), dtype=float)
    for i in range(N):
        proba_matrix[i,i]=0
        for j in range(i+1,N):
             proba_matrix[i, j]=proba_matrix[j,i] = max(W[i],W[j])*min(W[i],W[j])**sig/dT[i,j]**alpha
    return proba_matrix

#%%%%

def goe_variance(t,N,sigma):
    P=proba_matrix1(0, t, N, sigma) #probability matrix with alpha=0 
    A = np.random.randn(N, N)*np.sqrt(P) #taking gaussians and multiplying by \sqrt{proba matrix}
    B= A/np.sqrt(N)
    D = np.real(np.linalg.eigvals(B))
    return D
#%%% Plotting eigenvalues of gaussian matrix
eigens_D=goe_variance(tv,Nv,sigv)
plt.hist(eigens_D, bins=80, ec='black',density = True,color='g',alpha=0.5)
plt.grid(True)
 #print(summaDAD,sumplomaSFP)
plt.legend([r'$\tilde{A}_{N,g}$ ESD'])
plt.title('size='+str(Nv)+ ',  alpha=0, ' + 'tau='+ str(tv+1)+ ', sigma='+str(sigv) )
plt.show()
#%% Old stuff
    
#%%
'Plotting the egenvalues'
eigs=eigenvalues_SFP_cent(alphav,tv,Nv,sigv,n_matricesv)[0].ravel()
#fig = plt.figure(figsize=(12, 7))
#ax = fig.add_subplot(111)

#lines = ax.hist(eigs, bins=20, edgecolor="k", label="Histogram")

#ax.legend(loc="best")
#ax.grid(True, zorder=-5)

'initialize a univariate kernel density estimator'
#indix=np.where((eigs>1.6) & (eigs<1.9))  # fit for tails
kde = sm.nonparametric.KDEUnivariate(eigs)
#kde = sm.nonparametric.KDEUnivariate(eigs[indix])
kde.fit()  # Estimate the densities
'Plot the histogram and the fit'
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111)

# Plot the histogram
ax.hist(
    #eigs[indix],
   eigs,
    bins=20,
    density=True,
    label="Histogram eigenvalues",
    zorder=5,
    edgecolor="k",
    alpha=0.6,
)
plt.title( 'size=' +str(Nv)+ ', alpha=' +str(alphav)+ ', tau='+str(tv+1)+', sigma='+str(sigv) )

# Plot the KDE as fitted using the default arguments
ax.plot(kde.support, kde.density, lw=3, label="KDE from samples", zorder=10)


ax.legend(loc="best")
ax.grid(True, zorder=-5)

'Plotting the survival of the density'
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111)

indices=np.where((kde.support>1.6) & (kde.support<2)) 
xs=kde.support[indices]
ax.plot(np.log(xs),np.log(kde.sf[indices]) , lw=3, label="y=log  μ(x,∞)")

fs=(-2*tv/max(1,sigv))*np.log(xs/2)

ax.plot(np.log(xs),fs,label='y=-2(τ-1)/max(sigv,1)log(x/2)')
ax.legend(loc="best")
plt.title( 'size=' +str(Nv)+ ', alpha=' +str(alphav)+ ', tau='+str(tv+1)+', sigma='+str(sigv) )
ax.grid(True, zorder=-5)




regression_model = LinearRegression()
regression_model.fit(np.log(xs/2).reshape(-1,1), fs)

slope = regression_model.coef_
print(slope)

regression_model.fit(np.log(xs).reshape(-1,1),np.log(kde.sf[indices]) )
slope = regression_model.coef_
print(slope) 
