# Correlation Filter Tracking

The MOSSE (Minimum Output Sum of Squared Error) correlation filter tracker is a popular algorithm for visual object tracking. It uses correlation filters in the frequency domain to estimate the position and scale of a target object in video frames. By optimizing filter coefficients during training, it maximizes correlation with the target and minimizes correlation with the background. The MOSSE tracker is efficient, robust to variations in object appearance, and has real-time performance. It finds applications in visual surveillance, action recognition, and human-computer interaction, although it may face challenges with severe occlusion or rapid appearance changes.

## Simplified MOSSE
The main idea behind correlation filters is to learn the filter so that it has high correlation response on the object and low response on the background. In the first frame t = 1, the filter H is constructed using the equation:

$$ \hat{\bar{H}} = \frac{\hat{G} \odot \hat{\bar{F}}}{\hat{F} \odot \hat{\bar{F}} + \lambda}, $$

where G is a 2-dimensional Gaussian function and F is feature patch (i.e., grayscale image patch centered at object location). Operation $\odot$ is a point-wise product, division is also calculated element-wise and the ¯ denotes complex-conjugate operator. Note that the ^ represents variable in Fourier domain i.e., $\hat{a} = \mathcal{F}(a)$. Fourier transform must be performed in 2-dimensions e.g., numpy.fft.fft2.

After the filter has been constructed it can be used to localize the target (i.e., t = 2, 3, 4, ...). The localization step is implemented using equation:

$$ R = \mathcal{F}^{-1}(\hat{\bar{H}} \odot \hat{F}), $$


where R represents 2-dimensional correlation response and new target location is defined as position of the maximum peak in the response and $\mathcal{F}^{-1}$ is the inverse Fourier transform.

Using constant filter H does not model the target well, especially when it is changing its appearance. That is the reason for online update of the filter and it is typically realized as exponential forgetting:

$$ \hat{\bar{H_t}}} = (1-\alpha) \hat{\bar{H_{t-1}}} + \alpha \hat{\bar{\tilde{H}}}. $$

The updated filter at frame t is denoted as $\hat{\bar{H}}_t$ and the filter from previous frame is denoted as $\hat{\bar{H}}_{t-1}$. Filter at the current frame, obtained with the first equation is denoted
as \hat{\bar{\tilde{H}}}. Also, an important parameter here is the update speed $\alpha$ (typically a low number i.e., 0.02, 0.1, ...).

https://github.com/R4d0slav/Computer-Vision/assets/60989050/afac0f87-3b6c-4ef7-9e41-0202f3bc737a

## MOSSE
The MOSSE Tracker is impelemnted from the original paper https://ieeexplore.ieee.org/abstract/document/5539960/. The correlation filter is updated (and constructed) in a different way, where the numerator and denominator of the filter are stored separately. Also, the localization step is implemented differently, too.

https://github.com/R4d0slav/Computer-Vision/assets/60989050/37b805a6-0ddf-4c35-b490-31427ce7f0d0

