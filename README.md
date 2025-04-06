We wish to plot the spectral distribution of a scale-free percolation model. The model is described as follows: 
We consider (W_i) a sequence of Pareto random variables for i in the range [1,N], where N=no. of vertices on a torus. 

Let r_ij = W_iW_j/dist(i,j)^a, where dist(i,j) is the torus distance between i and j, and 0<a<1 a parameter. Then, we connect i and j with probabililty p_ij = min(W_iW_j/dist(i,j)^a,1). 

So, we plot eigenvalues of a matrix A with entries A(i,j)=A(j,i) distributed as Bernoulli(p_ij). 

We also show with simulations that this has similar spectrum to a matrix B = DGD, where D is a diagonal matrix with D(i,i) = sqrt{W_i}, and G(i,j)=G(j,i) is normally distributed with mean 0 and variance 1/dist(i,j)^a. 

Later on, we try estimating the tail of this spectral distribution, which theoretically has a power law decay. 
