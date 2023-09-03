# Advanced Vehicle Control with Model Predictive Control (MPC)
This repository delves into the realm of applying Model Predictive Control (MPC) to a f1tenth car, a vital control technique that leverages optimization to forecast and mitigate system behavior over a finite horizon. The primary objective is to understand and implement MPC to direct a vehicle to adhere closely to a predefined trajectory.

# Model Predictive Control (MPC) for Vehicle Control

## Overview

This project revolves around the implementation and understanding of Model Predictive Control (MPC). MPC is an advanced method of process control that utilizes optimization to achieve control objectives, mainly for multi-input, multi-output systems.

## Core Components

### Convex Optimization

The foundation of MPC relies on convex optimization techniques to predict and choose an optimized control action sequence over a future horizon.

### Linearization & Discretization

To formulate the MPC problem, it's essential to understand linearization and discretization techniques. This project utilized a kinematic model of a vehicle, linearized around specific points, and discretized using the Forward Euler method.

### State and Dynamics Model

The state space for our vehicle model is represented by position, orientation, and velocity. The inputs are acceleration and steering angle. With these, the kinematic equations that describe the vehicle's motion are constructed.

### Objective Function

The aim of the optimization is three-fold:
- Minimize the deviation of the vehicle from a given reference trajectory.
- Minimize the influence of the control inputs.
- Minimize the difference between successive control inputs.

### Constraints

To ensure that the optimization solution is feasible:
- It adheres to the vehicle's dynamics.
- It starts from the current vehicle state.
- It does not exceed the vehicle's physical limitations.

### Reference Trajectory

A trajectory with associated velocities for each waypoint is designed. This serves as a reference for the MPC to track.

### Optimization Setup

Leveraging Python's CVXPY library and the OSQP solver, the MPC optimization problem is constructed and solved. The code primarily requires defining the objective function and constraints for the MPC.

### Visualization

A critical aspect of this project was visualizing the current segment of the reference path and the MPC's predicted trajectory, aiding in debugging and providing insights.

![image](https://github.com/Saibernard/Advanced-Autonomous-Vehicle-Control-with-Model-Predictive-Control-MPC/assets/112599512/a637f64e-e827-4a80-8cd9-c71731e1c2fc)


## Simulation results in Rviz:

https://www.youtube.com/watch?v=ugK-iorKzo0

## Conclusion

This project not only showcases the MPC's theoretical aspects but also offers hands-on experience in implementing an MPC for vehicle control, a crucial element in autonomous vehicle systems. The result is a car tracking waypoints effectively in a simulated environment, demonstrating the power and precision of MPC in action.

