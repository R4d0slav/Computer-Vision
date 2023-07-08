<center><h1>Optical Flow</h1></center>

<b>Optical flow</b> is a fundamental concept in computer vision that aims to estimate the motion of objects within a sequence of images or a video. It provides a dense representation of the motion field by assigning a velocity vector to each pixel in an image, indicating the direction and magnitude of its movement over time. Optical flow has various applications in computer vision, including motion tracking, object detection and recognition, video stabilization, and autonomous navigation systems.

The estimation of optical flow is challenging due to the inherent ambiguity and complexity of motion patterns in real-world scenes. It involves determining the displacement of image features between consecutive frames and requires addressing issues such as occlusions, brightness changes, texture variations, and noise.

Two widely used optical flow methods are the Lucas-Kanade and Horn-Schunck algorithms. The <b>Lucas-Kanade</b> algorithm is a local method that assumes small motion between frames and computes the optical flow for each pixel independently. It formulates the optical flow estimation as a least squares problem, solving for the velocity vector that minimizes the discrepancy between the intensity values of a pixel's neighborhood in two consecutive frames.
The displacement vectors <i>u</i> and <i>v</i> for given input images $I_1$ and $I_2$ are calculated with:

$$ u = -\frac{\sum_N{I^2_y} \sum_N{I_xI_t} - \sum_N{I_xI_y} \sum_N{I_yI_t}}{D}, \quad -\frac{\sum_N{I^2_x} \sum_N{I_yI_t} - \sum_N{I_xI_y} \sum_N{I_xI_t}}{D}, $$

where N denotes the neighborhood of the pixel (usually a 3×3 pixel region), $I_x, I_y$ denote the two spatial derivatives (the pixel-wise average image derivatives of the first and the second image in x and y direction), and $I_t$ denotes the temporal derivative $I_2-I_1$. The D is the determinant of a covariance matrix that is defined as:

$$ D = \sum_N{I^2_x} \sum_N{I^2_y} - (\sum_N{I_xI_y})^2. $$

On the other hand, the <b>Horn-Schunck</b> algorithm is a global method that imposes smoothness constraints on the motion field. It assumes that neighboring pixels have similar motion and solves a partial differential equation to obtain a smooth motion field across the entire image. This algorithm provides a dense optical flow estimation by propagating the motion constraints globally.
For given input images $I_1$ and $I_2$ the resulting formulas for displacement vectors u and v are defined iteratively as:

$$ u = u_a - I_x \frac{P}{D}, \quad v = v_a - I_y \frac{P}{D}, $$

where the $u_a$ and $v_a$ are the iterative corrections to the displacement estimate, defined by convolving the corresponding component with a "residual Laplacian kernel":

$$ u_a = u \cdot L_d, \quad v_a = v \cdot L_d, \quad L_d = \begin{bmatrix}
0 & \frac{1}{4} & 0 \\
\frac{1}{4} & 0 & \frac{1}{4} \\
0 & \frac{1}{4} & 0 
\end{bmatrix}.
$$

The $I_x$ and $I_y$ denote the pixel-wise average image derivatives for the first and the second image in x and y direction and the <i>P</i> and <i>D</i> terms are defined as:

$$ P = I_xu_a + I_yv_a + I_t, \quad D = \lambda + I^2_x + I^2_y, $$

where $I_t$ denotes the time derivative $I_2-I_1$. The initial estimates for u and v are typically set to 0 and are then iteratively improved.

Both the Lucas-Kanade and Horn-Schunck methods have their strengths and weaknesses. The Lucas-Kanade algorithm is computationally efficient and performs well in scenarios with small displacements, making it suitable for real-time applications. However, it may struggle with large displacements and fails to handle occlusions and textureless regions effectively. On the other hand, the Horn-Schunck algorithm is more robust to noise and can handle larger displacements, but it is computationally expensive and may produce oversmoothed results.


https://github.com/R4d0slav/Computer-Vision/assets/60989050/5abef732-afb8-4127-b27e-b7513e8aaa35

