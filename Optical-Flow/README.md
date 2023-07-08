<center><h1>Optical Flow</h1></center>

<b>Optical flow</b> is a fundamental concept in computer vision that aims to estimate the motion of objects within a sequence of images or a video. It provides a dense representation of the motion field by assigning a velocity vector to each pixel in an image, indicating the direction and magnitude of its movement over time. Optical flow has various applications in computer vision, including motion tracking, object detection and recognition, video stabilization, and autonomous navigation systems.

The estimation of optical flow is challenging due to the inherent ambiguity and complexity of motion patterns in real-world scenes. It involves determining the displacement of image features between consecutive frames and requires addressing issues such as occlusions, brightness changes, texture variations, and noise.

Two widely used optical flow methods are the Lucas-Kanade and Horn-Schunck algorithms. The <b>Lucas-Kanade</b> algorithm is a local method that assumes small motion between frames and computes the optical flow for each pixel independently. It formulates the optical flow estimation as a least squares problem, solving for the velocity vector that minimizes the discrepancy between the intensity values of a pixel's neighborhood in two consecutive frames.

$$ u = - \frac{\sum_N{I^2_y} \sum_N{I_xI_t} - \sum_N{I_xI_y} \sum_N{I_yI_t}}{D} $$


On the other hand, the <b>Horn-Schunck</b> algorithm is a global method that imposes smoothness constraints on the motion field. It assumes that neighboring pixels have similar motion and solves a partial differential equation to obtain a smooth motion field across the entire image. This algorithm provides a dense optical flow estimation by propagating the motion constraints globally.

Both the Lucas-Kanade and Horn-Schunck methods have their strengths and weaknesses. The Lucas-Kanade algorithm is computationally efficient and performs well in scenarios with small displacements, making it suitable for real-time applications. However, it may struggle with large displacements and fails to handle occlusions and textureless regions effectively. On the other hand, the Horn-Schunck algorithm is more robust to noise and can handle larger displacements, but it is computationally expensive and may produce oversmoothed results.


https://github.com/R4d0slav/Computer-Vision/assets/60989050/5abef732-afb8-4127-b27e-b7513e8aaa35

