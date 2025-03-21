Bayesian Force Field Optimization with Local Gaussian Process Molecular Dynamics (LGPMD)

Overview

This repository provides source code for building Local Gaussian Process surrogate models designed to accelerate Bayesian optimization of force fields. LGPMD can be adapted seamlessly to model various quantities of interest in chemical and physical systems by adjusting training/test datasets, Gaussian Process (GP) mean functions, and kernel functions accordingly.

Features

Flexible surrogate modeling applicable to diverse chemical and physical observables.

Easy integration with Bayesian inference frameworks for parameter optimization.

Modular implementation allowing customized GP mean and kernel definitions.

Installation

Dependencies

emcee

pytorch

scipy

multiprocessing (standard library)

Install all required Python packages using pip:

pip install emcee torch scipy

Version History

v2.0 (Current Version)

Enhanced hyperparameter optimization via Bayesian hyperposterior inference.

Improved stability and performance compared to v1.0.

v1.0 (Deprecated)

Initial release featuring Local Gaussian Process surrogate modeling for radial distribution functions (RDFs) in monatomic fluids.

Hyperparameters trained using leave-one-out cross-validation on log marginal likelihood maximization over a fixed hyperparameter grid.

Known Issues: Parallelization with dask may cause errors; users are advised to upgrade to v2.0.

Usage

To get started, modify the training and testing datasets along with GP mean and kernel functions in the provided scripts to match your specific application. Example scripts demonstrating radial distribution function modeling are included.

License

This project is licensed under the MIT License. See LICENSE for details.

Contact

For questions, issues, or contributions, please open a GitHub issue or submit a pull request.



## Acknowledgement
The source code development was supported by the National Science Foundation under award number CBET-1847340. Developed at the University of Utah, Department of Chemical Engineering by Brennon Shanks, Harry Sullivan and Michael P. Hoepfner.  
