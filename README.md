# Bayesian Force Field Optimization with Local Gaussian Process Molecular Dynamics (LGPMD)
 
## General Information
Source code for constructing local Gaussian process surrogate models for accelerated Bayesian force field optimization. The code is adaptable to any quantity-of-interest in chemistry in physics applications with appropriate updates to training and test data matrices as well as Gaussian process mean and kernel functions. 

## Required Software

emcee
pytorch
scipy
multiprocessing

## Version History
v1.0 - Local Gaussian process surrogate model for radial distribution functions of monatomic fluids. Hyperparameter training performed using leave-one-out log marginal likelihood maximization over a fixed hyperparameter space.

v2.0 - Updated hyperparameter optimization using Bayesian optimization of the hyperposterior with a leave-one-out log marginal likelihood.

## Acknowledgement
The source code development was supported by the National Science Foundation under award number CBET-1847340. Developed at the University of Utah, Department of Chemical Engineering by Brennon Shanks, Harry Sullivan and Michael P. Hoepfner.  
