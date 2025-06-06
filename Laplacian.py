#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 13:58:26 2025

@author: malhotran
"""

import numpy as np
from matplotlib import pyplot as plt
#from numba import njit
from scipy.stats import pareto
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
def proba_matrix(alpha,t,N):
    dT = distancematrix(N)
    #Uncomment below for same W
    #np.random.seed(10)
    #W_unif = np.random.randint(1,15,N) 
    #m_1 = np.sum(W_unif[i] for i in range(N)) 
    W = pareto.rvs(t, size=N) #sampling Paretos with P(W>x)=1/x**{t} so t=tau-1 in our language
    m_1 = np.sum(W[i] for i in range(N))
    proba_matrix =  np.empty(shape=(N, N), dtype=float)
    for i in range(N):
        proba_matrix[i,i]=0
        for j in range(i+1,N):
             proba_matrix[i, j]=proba_matrix[j,i] = min((W[i]*W[j])/(dT[i,j]**alpha),1)
             #proba_matrix[i, j]=proba_matrix[j,i] = min(1/(dT[i,j]**alpha),1)
             #proba_matrix[i, j]=proba_matrix[j,i] = min(1,W[i]*W[j]/(m_1 + W[i]*W[j]))
             #proba_matrix[i,j]=proba_matrix[j,i] = 3.5/N 
    return proba_matrix

"Construction of centered adjacency matrix"

def cen_adj_matrix(alpha,t,N):
    adj_matrix = np.empty(shape=(N, N), dtype=float)
    pro_matrix = proba_matrix(alpha,t,N)
    for i in range(N):
        adj_matrix[i,i]=0
        for j in range(i+1,N):
            realisation = np.random.binomial(1,pro_matrix[i, j]) ## I compute the Bernoulli's in each entry as Binomial(1,p)
            adj_matrix[i, j]=adj_matrix[j, i]=realisation  - pro_matrix[i, j] 
    return adj_matrix
#%% Calculating eigenvalues of the centered matrix

"Construction of Laplacians (A-D)" 
"Centered Laplacian"

def cen_Laplacian(alpha,t,N):
    lap_matrix = np.empty(shape=(N,N), dtype=float)
    adj = cen_adj_matrix(alpha,t,N) 
    for i in range(N):
        lap_matrix[i,i]=0
        for j in range(i+1,N):
            lap_matrix[i, j]=lap_matrix[j, i]=adj[i, j] ##off diagonal terms are centered adjacency terms
        lap_matrix[i,i] = -1*(np.sum(lap_matrix[i,k] for k in range(N)))  #diagonal as negative row sum
    return lap_matrix 

"Gaussianised Laplacian"
def gauss_Laplacian(alpha,t,N):
    lap_matrix = np.empty(shape=(N,N), dtype=float) 
    W = pareto.rvs(t, size=N)
    dT = distancematrix(N) 
    for i in range(N):
        lap_matrix[i,i]=0
        for j in range(i+1,N):
            lap_matrix[i, j]=lap_matrix[j, i]=np.sqrt((W[i]*W[j])/dT[i,j])*np.random.normal(0,1)
        lap_matrix[i,i] = -1*(np.sum(lap_matrix[i,k] for k in range(N))) 
    return lap_matrix
    
"Decoupled Laplacian with alpha=0" 
def decoup_Laplacian(t,N):
    lap_matrix = np.empty(shape=(N,N), dtype=float)
    W = pareto.rvs(t, size=N)
    for i in range(N):
        for j in range(i+1,N):
            lap_matrix[i, j]=lap_matrix[j, i]=np.sqrt(W[i]*W[j])*np.random.normal(0,1)
        lap_matrix[i, i] = -np.sqrt(W[i])*np.sqrt((t-1/t-2))*np.random.normal(0,1)
    return lap_matrix 
    
def eigenvalues_SFP_laplacian(N,n_matrices,Laplacian):
    eig = np.zeros([n_matrices, N])
    summa = np.empty(shape=(n_matrices, ), dtype=float)
    for i in range(n_matrices):
        eig[i, :] = LA.eigvalsh(Laplacian) 
        summa[i] = np.sum(eig[i, :]**2)/N #Tr(A^2)=1/N sum lambda_i^2
    normalizing=np.sqrt(np.mean(summa))
    eig=eig/normalizing 
    return eig, summa


#%%% Plotting the eigenvalues of DAD and SFP
def plot_random_matrix_eigenvalues(alpha, t , N, n_matrices):
    eig_cen, summa_cen = eigenvalues_SFP_laplacian(N,n_matrices,cen_Laplacian(alpha, t, N))
    #eig_gauss, summa_gauss = eigenvalues_SFP_laplacian(N,n_matrices,gauss_Laplacian(alpha, t, N))
    #eig_decoup, summa_decoup = eigenvalues_SFP_laplacian(N,n_matrices,decoup_Laplacian(t, N))
    plt.figure(figsize = (10, 5))
    plt.title("ESD")
    plt.hist(eig_cen.ravel(), bins=80, ec='black',density = True,color='royalblue',alpha=0.5)
    #plt.hist(eig_gauss.ravel(), bins=80, ec='black',density = True,color='g',alpha=0.5)
    #plt.hist(eig_decoup.ravel(), bins=80, ec='black',density = True,color='r',alpha=0.5)
    plt.grid(True)
    #print(summaDAD,sumplomaSFP)
    plt.legend(["Eigenvalue distribution", "Decoupled Laplacian, alpha=0"])
    plt.title('size='+str(N)+ ', alpha=' +str(alpha)+ ', tau='+ str(t+1) )
    #plt.ylim(0,1) 
    plt.savefig('SFPLaplacian_case2.png') 
    plt.show()
    
#%%%%% T'Setting the parameters'
alphav=1.5
tv=1.5 
Nv=2000
sigv=0.2
n_matricesv=1
#%%'Calculating sf'

plot_random_matrix_eigenvalues(alphav, tv, Nv, n_matricesv)




