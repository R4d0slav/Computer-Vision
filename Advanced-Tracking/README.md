# Advanced Tracking

Object tracking plays a crucial role in computer vision applications, enabling the monitoring and analysis of objects in dynamic environments. 
One popular approach for object tracking is the use of advanced techniques such as particle filters.

<b>Particle Filters</b>, also known as Bayesian Filters or Sequential Monte Carlo Methods, are stochastic algorithms that estimate the state of an object by representing it as a set of particles. 
Each particle corresponds to a possible hypothesis of the object's state, such as its position, velocity, or orientation. 
These particles are propagated through time according to a <b>motion model</b>, and their weights are updated based on their likelihood with respect to the observed measurements. 
By iteratively resampling and updating the particles, the particle filter converges to an accurate representation of the object's state.

## Motion models
<b>Random Walk (RW) Model</b>:
- The RW model assumes that the object's motion is governed by random displacements at each time step.
- In this model, the object's state is represented by its position only.
- The state transition matrix (F) for the RW model is set to zero, indicating no change in the state between consecutive time steps.
- The observation matrix (H) is an identity matrix, meaning that the observed measurements directly correspond to the object's position.
- The process noise covariance matrix (Q) represents the uncertainty in the random displacements.
- The RW model is suitable for scenarios where the object's motion does not exhibit a clear pattern or follows random movements.

$$
\begin{align*}
X_{state} = \begin{bmatrix}
    x\\
    y\\
    \dot{x}\\
    \dot{y}
\end{bmatrix}
F = \begin{bmatrix}
    0 & 0 & 1 & 0 \\
    0 & 0 & 0 & 1 \\
    0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0
\end{bmatrix}
\phi = \begin{bmatrix}
    1 & 0 & \Delta T & 0\\
    0 & 1 & 0 & \Delta T\\
    0 & 0 & 1 & 0\\
    0 & 0 & 0 & 1
\end{bmatrix}
L = \begin{bmatrix}
    0 & 0 \\
    0 & 0 \\
    1 & 0 \\
    0 & 1
\end{bmatrix}
H = \begin{bmatrix}
    1 & 0 & 0 & 0 \\
    0 & 1 & 0 & 0
\end{bmatrix}
Q = \begin{bmatrix}
    \Delta T q & 0 & \Delta T q & 0 \\
    0 & \Delta T q & 0 & \Delta T q \\
    \Delta T q & 0 & \Delta T q & 0 \\
    0 & \Delta T q & 0 & \Delta T q
\end{bmatrix}
\end{align*}
$$


<b>Nearly Constant Velocity (NCV) Model</b>:
- The NCV model assumes that the object's motion follows a nearly constant velocity over time.
- In this model, the object's state is represented by its position and velocity.
- The state transition matrix (F) represents the linear dynamics of the object's position and velocity over time.
- The observation matrix (H) is similar to the RW model, where the observed measurements directly correspond to the object's position.
- The process noise covariance matrix (Q) represents the uncertainty in the object's acceleration, accounting for small variations from constant velocity.
- The NCV model is suitable for scenarios where the object's motion can be approximated as nearly constant velocity with minor variations.

$$
\small
\begin{align*}
X_{state} = \begin{bmatrix}
    x\\
    y\\
    \dot{x}\\
    \dot{y}
\end{bmatrix}
F = \begin{bmatrix}
    0 & 0 & 1 & 0 \\
    0 & 0 & 0 & 1 \\
    0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0
\end{bmatrix}
\phi = \begin{bmatrix}
    1 & 0 & \Delta T & 0\\
    0 & 1 & 0 & \Delta T\\
    0 & 0 & 1 & 0\\
    0 & 0 & 0 & 1
\end{bmatrix}
L = \begin{bmatrix}
    0 & 0 \\
    0 & 0 \\
    1 & 0 \\
    0 & 1
\end{bmatrix}
H = \begin{bmatrix}
    1 & 0 & 0 & 0 \\
    0 & 1 & 0 & 0
\end{bmatrix}
Q = \begin{bmatrix}
    \Delta T q & 0 & \Delta T q & 0 \\
    0 & \Delta T q & 0 & \Delta T q \\
    \Delta T q & 0 & \Delta T q & 0 \\
    0 & \Delta T q & 0 & \Delta T q
\end{bmatrix}
\end{align*}
$$
  
<b>Nearly Constant Acceleration (NCA) Model</b>:
- The NCA model assumes that the object's motion follows a nearly constant acceleration over time.
- In this model, the object's state is represented by its position, velocity, and acceleration.
- The state transition matrix (F) represents the linear dynamics of the object's position, velocity, and acceleration over time.
- The observation matrix (H) is similar to the NCV model, where the observed measurements directly correspond to the object's position.
- The process noise covariance matrix (Q) represents the uncertainty in the object's jerk (derivative of acceleration), accounting for small variations from constant acceleration.
- The NCA model is suitable for scenarios where the object's motion can be approximated as nearly constant acceleration with minor variations.

$$
\tiny{
\begin{align*}
X_{state} = \begin{bmatrix}
    x\\
    y\\
    \dot{x}\\
    \dot{y}\\
    \ddot{x}\\
    \ddot{y}
\end{bmatrix}
F = \begin{bmatrix}
    0 & 0 & 1 & 0 & 0 & 0\\
    0 & 0 & 0 & 1 & 0 & 0\\
    0 & 0 & 0 & 0 & 1 & 0\\
    0 & 0 & 0 & 0 & 0 & 1\\
    0 & 0 & 0 & 0 & 0 & 0\\
    0 & 0 & 0 & 0 & 0 & 0
\end{bmatrix}
\phi = \begin{bmatrix}
    1 & 0 & \Delta T & 0 & \frac{\Delta T^2}{2} & 0\\
    0 & 1 & 0 & \Delta T & 0 & \frac{\Delta T^2}{2}\\
    0 & 0 & 1 & 0 & \Delta T & 0\\
    0 & 0 & 0 & 1 & 0 & \Delta T\\
    0 & 0 & 0 & 0 & 1 & 0\\
    0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix}
L = \begin{bmatrix}
    0 & 0 \\
    0 & 0 \\
    0 & 0 \\
    0 & 0 \\
    1 & 0 \\
    0 & 1
\end{bmatrix}
Q = \begin{bmatrix}
    \frac{\Delta T q}{4} & 0 & \frac{\Delta T q}{2} & 0 & \frac{\Delta T q}{2} & 0 \\
    0 & \frac{\Delta T q}{4} & 0 & \frac{\Delta T q}{2} & 0 & \frac{\Delta T q}{2} \\
    \frac{\Delta T q}{2} & 0 & \Delta T q & 0 & \Delta T q & 0 \\
    0 & \frac{\Delta T q}{2} & 0 & \Delta T q & 0 & \Delta T q \\
    \frac{\Delta T q}{2} & 0 & \Delta T q & 0 & \Delta T q & 0 \\
    0 & \frac{\Delta T q}{2} & 0 & \Delta T q & 0 & \Delta T q
\end{bmatrix}
H = \begin{bmatrix}
    1 & 0 & 0 & 0 & 0 & 0 \\
    0 & 1 & 0 & 0 & 0 & 0
\end{bmatrix}
\end{align*}
}
$$

https://github.com/R4d0slav/Computer-Vision/assets/60989050/fd9d51c1-1934-400b-9c6b-f45489a18404

