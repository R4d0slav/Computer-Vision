# Advanced Tracking

Object tracking plays a crucial role in computer vision applications, enabling the monitoring and analysis of objects in dynamic environments. 
One popular approach for object tracking is the use of advanced techniques such as particle filters.

<b>Particle Filters</b>, also known as Bayesian Filters or Sequential Monte Carlo Methods, are stochastic algorithms that estimate the state of an object by representing it as a set of particles. 
Each particle corresponds to a possible hypothesis of the object's state, such as its position, velocity, or orientation. 
These particles are propagated through time according to a <b>motion model</b>, and their weights are updated based on their likelihood with respect to the observed measurements. 
By iteratively resampling and updating the particles, the particle filter converges to an accurate representation of the object's state.

## Motion models
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




https://github.com/R4d0slav/Computer-Vision/assets/60989050/fd9d51c1-1934-400b-9c6b-f45489a18404

