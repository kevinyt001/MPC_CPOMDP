# Stochastic Predictive Control Partially Observable Markov Decision Process

Stochastic control with POMDP methods

This project aims to implement a solver of stochastic control for POMDP methods with easy-to-use interface while keeping codes readable. This project implements a solver for the framework developed by N. Li in University of Michigan, which you can find [here](https://asmedigitalcollection.asme.org/dynamicsystems/article/726497/Stochastic-Predictive-Control-for-Partially). The codes are generated taking inspiration from `AI-Toolbox` written by E. Bargiacchi, which you can find [here](https://github.com/Svalorzen/AI-Toolbox), and from the `pomdp-solve` software written by A. R. Cassandra, which you can find [here](http://www.pomdp.org/code/index.shtml). 

## COMPILE INSTRUCTIONS

###### Dependencies

To compile the MPC-POMDP in C++, you need:

- Full C++17 support is now required (**at least g++-7**)
- [cmake](http://www.cmake.org/) >= 3.9
- the [boost library](http://www.boost.org/) >= 1.54
- the [Eigen 3.3 library](http://eigen.tuxfamily.org/index.php?title=Main_Page).
- the [NLOPT Library](https://nlopt.readthedocs.io/)

###### Compiling

After the installation of dependencies, follow these steps in the project directory:

```
mkdir build
cd build
cmake ..
make
```

## EXECUTING

To run with the default autonomous overtaking scenarios, just run 
this executable.

```
cd ../bin
./run_overtake
```

you may check the default POMDP file:

```
more ../input/overtake.POMDP
```
