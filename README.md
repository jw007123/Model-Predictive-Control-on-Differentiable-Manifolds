# Model Predictive Control on Differentiable Manifolds
An implementation of [Model Predictive Control for Trajectory Tracking on Differentiable Manifolds](https://arxiv.org/pdf/2106.15233.pdf). The header-only library provides a generic, nonlinear MPC solver class and solutions for particular platforms can be implemented as derived classes. Realtime performance is achieved through the use of templates, minimal run-time memory allocation and exceptional QP solving courtesy of [OSQP](https://osqp.org/) and [Eigen 3](https://eigen.tuxfamily.org/index.php?title=Main_Page)-- the only external dependencies. 

All non-example code is contained within *Include/* as two header files: *'NonLinearSolver.h'* and *'Literals.h'*, where the latter is nothing more than Rust-style typedefs of basic types. The two examples (a quadcopter and a simple, 2D land vehicle) can be found in *Examples/* and provide information on how to interface with the solver for different platform dynamics models. Finally, there is a set of unit-tests in *Tests/* that show how to implement the basic MPC loop.

| ![fig4.png](https://i.imgur.com/oaYaOA0.png) | 
|:--:| 
| A visual interpration of the MPC problem. Fig (4) in the attached paper. |
