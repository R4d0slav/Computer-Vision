import math

import numpy as np
import cv2


def gausssmooth(img, sigma):
    x = np.array(list(range(math.floor(-3.0 * sigma + 0.5), math.floor(3.0 * sigma + 0.5) + 1)))
    G = np.exp(-x**2 / (2 * sigma**2))
    G = G / np.sum(G)
    return cv2.sepFilter2D(img, -1, G, G)

def generate_responses_1():
    responses = np.zeros((100, 100), dtype=np.float32)
    responses[70, 50] = 1
    responses[50, 70] = 0.5
    return gausssmooth(responses, 10)

def get_patch(img, center, sz):
    # crop coordinates
    x0 = round(int(center[0] - sz[0] / 2))
    y0 = round(int(center[1] - sz[1] / 2))
    x1 = int(round(x0 + sz[0]))
    y1 = int(round(y0 + sz[1]))
    # padding
    x0_pad = max(0, -x0)
    x1_pad = max(x1 - img.shape[1] + 1, 0)
    y0_pad = max(0, -y0)
    y1_pad = max(y1 - img.shape[0] + 1, 0)

    # Crop target
    if len(img.shape) > 2:
        img_crop = img[y0 + y0_pad:y1 - y1_pad, x0 + x0_pad:x1 - x1_pad, :]
    else:
        img_crop = img[y0 + y0_pad:y1 - y1_pad, x0 + x0_pad:x1 - x1_pad]

    im_crop_padded = cv2.copyMakeBorder(img_crop, y0_pad, y1_pad, x0_pad, x1_pad, cv2.BORDER_REPLICATE)

    # crop mask tells which pixels are within the image (1) and which are outside (0)
    m_ = np.ones((img.shape[0], img.shape[1]), dtype=np.float32)
    crop_mask = m_[y0 + y0_pad:y1 - y1_pad, x0 + x0_pad:x1 - x1_pad]
    crop_mask = cv2.copyMakeBorder(crop_mask, y0_pad, y1_pad, x0_pad, x1_pad, cv2.BORDER_CONSTANT, value=0)
    return im_crop_padded, crop_mask

def create_epanechnik_kernel(width, height, sigma):
    # make sure that width and height are odd
    w2 = int(math.floor(width / 2))
    h2 = int(math.floor(height / 2))

    [X, Y] = np.meshgrid(np.arange(-w2, w2 + 1), np.arange(-h2, h2 + 1))
    X = X / np.max(X)
    Y = Y / np.max(Y)

    kernel = (1 - ((X / sigma)**2 + (Y / sigma)**2))
    kernel = kernel / np.max(kernel)
    kernel[kernel<0] = 0
    return kernel

def extract_histogram(patch, nbins, weights=None):
    # Note: input patch must be a BGR image (3 channel numpy array)
    # convert each pixel intensity to the one of nbins bins
    channel_bin_idxs = np.floor((patch.astype(np.float32) / float(255)) * float(nbins - 1))
    # calculate bin index of a 3D histogram
    bin_idxs = (channel_bin_idxs[:, :, 0] * nbins**2  + channel_bin_idxs[:, :, 1] * nbins + channel_bin_idxs[:, :, 2]).astype(np.int32)

    # count bin indices to create histogram (use per-pixel weights if given)
    if weights is not None:
        histogram_ = np.bincount(bin_idxs.flatten(), weights=weights.flatten())
    else:
        histogram_ = np.bincount(bin_idxs.flatten())
    # zero-pad histogram (needed since bincount function does not generate histogram with nbins**3 elements)
    histogram = np.zeros((nbins**3, 1), dtype=histogram_.dtype).flatten()
    histogram[:histogram_.size] = histogram_
    return histogram

def backproject_histogram(patch, histogram, nbins):
    # Note: input patch must be a BGR image (3 channel numpy array)
    # convert each pixel intensity to the one of nbins bins
    channel_bin_idxs = np.floor((patch.astype(np.float32) / float(255)) * float(nbins - 1))
    # calculate bin index of a 3D histogram
    bin_idxs = (channel_bin_idxs[:, :, 0] * nbins**2  + channel_bin_idxs[:, :, 1] * nbins + channel_bin_idxs[:, :, 2]).astype(np.int32)

    # use histogram us a lookup table for pixel backprojection
    backprojection = np.reshape(histogram[bin_idxs.flatten()], (patch.shape[0], patch.shape[1]))
    return backprojection


def create_cosine_window(target_size):
    # target size is in the format: (width, height)
    # output is a matrix of dimensions: (width, height)
    return cv2.createHanningWindow((target_size[0], target_size[1]), cv2.CV_32F)

def create_gauss_peak(target_size, sigma):
    # target size is in the format: (width, height)
    # sigma: parameter (float) of the Gaussian function
    # note that sigma should be small so that the function is in a shape of a peak
    # values that make sens are approximately from the interval: ~(0.5, 5)
    # output is a matrix of dimensions: (width, height)
    w2 = math.floor(target_size[0] / 2)
    h2 = math.floor(target_size[1] / 2)
    [X, Y] = np.meshgrid(np.arange(-w2, w2 + 1), np.arange(-h2, h2 + 1))
    G = np.exp(-X**2 / (2 * sigma**2) - Y**2 / (2 * sigma**2))
    G = np.roll(G, (-h2, -w2), (0, 1))
    return G


def kalman_step(A, C, Q, R, y, x, V):
    # INPUTS:
    # A - the system matrix
    # C - the observation matrix
    # Q - the system covariance
    # R - the observation covariance
    # y(:)   - the observation at time t
    # x(:) - E[X | y(:, 1:t-1)] prior mean
    # V(:,:) - Cov[X | y(:, 1:t-1)] prior covariance
    # OUTPUTS (where X is the hidden state being estimated)
    # xnew(:) =   E[ X | y(:, 1:t) ]
    # Vnew(:,:) = Var[ X(t) | y(:, 1:t) ]
    # VVnew(:,:) = Cov[ X(t), X(t-1) | y(:, 1:t) ]
    # loglik = log P(y(:,t) | y(:,1:t-1)) log-likelihood of innovation

    xpred = np.matmul(A, x)
    Vpred = np.matmul(np.matmul(A, V), A.transpose()) + Q

    e = y - np.matmul(C, xpred)  # error (innovation)
    S = np.matmul(np.matmul(C, Vpred), C.transpose()) + R
    ss = max(V.shape)
    loglik = gaussian_prob(e, np.zeros((1, e.size), dtype=np.float32), S, use_log=True)
    K = np.linalg.lstsq(S.transpose(), np.matmul(Vpred, C.transpose()).transpose(), rcond=None)[0].transpose()  # Kalman gain matrix
    # If there is no observation vector, set K = zeros(ss).
    xnew = xpred + np.matmul(K, e)
    Vnew = np.matmul((np.eye(ss) - np.matmul(K, C)), Vpred)
    VVnew = np.matmul(np.matmul((np.eye(ss) - np.matmul(K, C)), A), V)

    return xnew, Vnew, loglik, VVnew

def gaussian_prob(x, m, C, use_log=False):
    # Evaluate multivariate Gaussian density
    # p(i) = N(X(:,i), m, C) where C = covariance matrix and each COLUMN of x is a datavector
    # p = gaussian_prob(X, m, C, 1) returns log N(X(:,i), m, C) (to prevents underflow).
    # If X has size dxN, then p has size Nx1, where N = number of examples

    if m.size == 1:
        x = x.flatten().transpose()

    d, N = x.shape

    m = m.flatten()
    M = np.reshape(m * np.ones(m.shape, dtype=np.float32), x.shape)
    denom = (2 * math.pi)**(d/2) * np.sqrt(np.abs(np.linalg.det(C)))
    mahal = np.sum(np.linalg.solve(C.transpose(), (x - M)) * (x - M))   # Chris Bregler's trick

    if np.any(mahal < 0):
        print('Warning: mahal < 0 => C is not psd')

    if use_log:
        p = -0.5 * mahal - np.log(denom)
    else:
        p = np.divide(np.exp(-0.5 * mahal), (denom + 1e-20))

    return p

def sample_gauss(mu, sigma, n):
    # sample n samples from a given multivariate normal distribution
    return np.random.multivariate_normal(mu, sigma, n)



# base class for tracker
class Tracker():
    def __init__(self, params):
        self.parameters = params

    def initialize(self, image, region):
        raise NotImplementedError

    def track(self, image):
        raise NotImplementedError


