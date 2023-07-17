# Computer Vision

This repository contains implementations of various computer vision tasks. Each task focuses on a specific aspect of computer vision, and the code provided can serve as a starting point for further research or development in the field. Below, a brief explanation of each task and its basic meaning is provided.

## Optical Flow

Optical flow refers to the pattern of apparent motion of objects, surfaces, and edges in a visual scene caused by the relative motion between the observer and the scene. It involves tracking the movement of pixels between consecutive frames of a video sequence. Optical flow estimation is crucial for tasks such as motion analysis, object tracking, and video stabilization. 
Two approaches for optical-flow are implemented: 
- <b>Lucas-Kanade</b>
- <b>Horn-Schunck</b>

https://github.com/R4d0slav/Computer-Vision/assets/60989050/5abef732-afb8-4127-b27e-b7513e8aaa35


## Mean-Shift Tracking

Mean-shift tracking is a non-parametric technique used for object tracking in computer vision. It involves locating an object in a video sequence by iteratively shifting a window's centroid towards the mode of the probability distribution of pixel intensities within that window. Mean-shift tracking is robust to variations in appearance, making it suitable for tracking objects with changing lighting conditions or occlusions.

https://github.com/R4d0slav/Computer-Vision/assets/60989050/4534d0a4-62f0-45d4-b10e-4ab9bcc63b73


## Discriminative Tracking (Correlation Filters)

Discriminative tracking using <b>correlation filters</b> is a popular technique for object tracking. It formulates the tracking problem as a discriminative classification task, where a classifier is trained to distinguish between the target object and the background. By correlating the classifier's response with the target template in a new frame, the tracker can estimate the target's position accurately.

https://github.com/R4d0slav/Computer-Vision/assets/60989050/0aab82ed-bba5-47c6-ac00-ed1d3bf44a15


## Advanced Tracking (Particle Filters)

Advanced tracking techniques, such as particle filters, are probabilistic methods used for tracking objects in videos. <b>Particle filters</b> represent the object's state using a set of particles, each associated with a probability weight. These particles are propagated and updated over time based on a motion model and observations from the video frames, allowing for robust and accurate tracking even in challenging scenarios with occlusions and appearance changes.

https://github.com/R4d0slav/Computer-Vision/assets/60989050/fd9d51c1-1934-400b-9c6b-f45489a18404


## Long-Term Tracking (SiamFC with Re-detection)

Long-term tracking refers to the ability to track objects across extended periods of time, even when they temporarily disappear from the camera's view. <b>SiamFC</b> (Siam Fully Convolutional) is a popular long-term tracking approach that employs a siamese network to learn a similarity measure between the target object and candidate regions in subsequent frames. <b>Re-detection</b> mechanisms are used to handle object reappearances, enabling the tracker to maintain accurate tracking over long durations.

https://github.com/R4d0slav/Computer-Vision/assets/60989050/a61f302e-f163-470b-9d2b-81be4730638a


## Getting Started

To get started with any of the tasks, please refer to the respective directories for detailed instructions on how to run the code and any dependencies required. You can find additional information and references in the individual task directories.

## Contributions

Contributions to the project are welcome! If you have any suggestions, bug reports, or would like to add more computer vision tasks, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE), which allows for both personal and commercial use. However, please refer to the license file for more information and to understand your rights and responsibilities when using this code.

