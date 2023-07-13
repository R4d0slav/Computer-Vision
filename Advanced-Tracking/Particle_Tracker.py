import time
import sympy as sp
from ex4_utils import *
import matplotlib.pyplot as plt
from sequence_utils import VOTSequence

import importlib


TOOLKIT = True
try:
    import sys
    sys.path.insert(0, '../toolkit-dir/utils')
    from tracker import Tracker
except ImportError as e:
    TOOLKIT = False
    pass


def init_model(model: str, dt: int, q_val: float, r_val: float):
    T, q, r = sp.symbols('T q r')

    R_i = sp.Matrix([[r, 0], 
                     [0, r]])
    
    if model == 'RW':

        H = sp.Matrix([[1, 0], 
                       [0, 1]])
        
        F = sp.Matrix([[0, 0],
                       [0, 0]])
                    
        L = sp.Matrix([[1, 0], 
                       [0, 1]])
        
    elif model == 'NCV':

        H = sp.Matrix([[1, 0, 0, 0], 
                       [0, 1, 0, 0]])
        
        F = sp.Matrix([[0, 0, 1, 0],
                       [0, 0, 0, 1],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0]])
            
        L = sp.Matrix([[0, 0], 
                       [0, 0], 
                       [1, 0],
                       [0, 1]]) 
                    
    elif model == 'NCA':

        H = sp.Matrix([[1, 0, 0, 0, 0, 0], 
                       [0, 1, 0, 0, 0, 0]])
        
        F = sp.Matrix([[0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0]])
            
        L = sp.Matrix([[0, 0], 
                       [0, 0], 
                       [0, 0], 
                       [0, 0],
                       [1, 0],
                       [0, 1]])
        
    else:
        raise Exception("Invalid model!")
    
    Fi = sp.exp(F*T)
    Fi = Fi.subs([(T, dt)])

    Q_i = sp.integrate((Fi*L)*q*(Fi*L).T, (T, 0, T))
    Q_i = Q_i.subs([(T, dt), (q, q_val)])
    R_i = R_i.subs([(r, r_val)])

    return np.array(Fi).astype(np.float64), np.array(H).astype(np.float64), np.array(Q_i).astype(np.float64), np.array(R_i).astype(np.float64)


class ParticleParams:
    def __init__(self):
        self.N = 150
        self.nbins = 16
        self.sigma = 1.0
        self.distance_sigma = 0.15
        self.alpha = 0.05
        self.scale_factor = 0.7

        self.model = 'NCV'
        self.q_rate = 0.1
        self.r = 1

class Particle_Tracker(Tracker):
    def __init__(self, parameters = ParticleParams()):
        self.parameters = parameters

    def name(self):
        return "Particle Tracker"
    
    def preprocess_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

        image_log = np.log(image + 1)

        mean = np.mean(image_log)
        std = np.std(image_log)
        image_norm = (image_log - mean) / std

        return image_norm


    def initialize(self, image, region):
        assert self.parameters.model in ['NCV', 'NCA', 'RW'], 'Invalid model!'

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        self.x, self.y, self.width, self.height = np.array(region).astype(int)

        self.width = int(self.width * self.parameters.scale_factor)
        self.height = int(self.height * self.parameters.scale_factor)

        self.width = self.width + 1 if self.width % 2 == 0 else self.width
        self.height = self.height + 1 if self.height % 2 == 0 else self.height

        patch, _ = get_patch(image, (self.x+int(self.width/2), self.y+int(self.height/2)), (self.width, self.height))

        self.k = create_epanechnik_kernel(patch.shape[0], patch.shape[1], self.parameters.sigma)
        self.k = self.k / self.k.sum()

        self.q = extract_histogram(patch, self.parameters.nbins, weights = self.k)
        self.q = self.q / self.q.sum()

        q = self.parameters.q_rate * min(self.width, self.height)
        self.A, self.C, self.Q_i, self.R_i = init_model(self.parameters.model, 1, q, self.parameters.r)
        
        state = [self.x+self.width/2, self.y+self.height/2]
        state.extend([0, 0] if self.parameters.model == 'NCV' else [0, 0, 0, 0]) if self.parameters.model != 'RW' else None

        noise = sample_gauss(np.zeros(self.Q_i.shape[0]), self.Q_i, self.parameters.N)
        self.particles = np.array([state + noise for noise in noise])
        self.weights = np.ones(self.parameters.N)


    def track(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        noise = sample_gauss(np.zeros(self.Q_i.shape[0]), self.Q_i, self.parameters.N)

        # Resample
        weights_norm = self.weights / np.sum(self.weights)
        weights_cumsumed = np.cumsum(weights_norm)
        rand_samples = np.random.rand(self.parameters.N, 1)
        sampled_idxs = np.digitize(rand_samples, weights_cumsumed)
        self.particles = self.particles[sampled_idxs.flatten() , :]

        # Move particles
        self.particles = np.transpose(np.matmul(self.A, np.transpose(self.particles))) + noise
        self.particles = np.clip(self.particles, [0, 0] + [-float('inf') for _ in range(2, len(self.particles[0]))], [image.shape[1]-1, image.shape[0]-1] + [float('inf') for _ in range(2, len(self.particles[0]))])

        # Recalculate weights
        patches = [get_patch(image, (particle[0], particle[1]), (self.width, self.height))[0] for particle in self.particles]
        histograms = np.array([extract_histogram(patch, self.parameters.nbins, weights=self.k) for patch in patches])
        histograms = histograms / histograms.sum(axis=1)[:, None]

        diffs = np.sqrt(np.sum((np.sqrt(histograms) - np.sqrt(self.q)) ** 2, axis=1)) / np.sqrt(2)
        self.weights = np.exp(-0.5*(diffs**2)/self.parameters.distance_sigma**2)

        weights_norm = self.weights/self.weights.sum()
        self.x = int(sum([self.particles[i, 0] * weights_norm[i] for i in range(self.parameters.N)]))
        self.y = int(sum([self.particles[i, 1] * weights_norm[i] for i in range(self.parameters.N)]))
        
        self.x = np.clip(self.x, 0, image.shape[1]-1)
        self.y = np.clip(self.y, 0, image.shape[0]-1)

        # Update model
        if self.parameters.alpha > 0:
            patch, _ = get_patch(image, (self.x, self.y), (self.width, self.height))
            q = extract_histogram(patch, self.parameters.nbins, weights = self.k)
            q = q / q.sum()
            self.q = (1-self.parameters.alpha)*self.q + self.parameters.alpha*q

        return [self.x - int(self.width/2), self.y - int(self.height/2), self.width/self.parameters.scale_factor, self.height/self.parameters.scale_factor] if TOOLKIT else ([self.x - int(self.width/2), self.y - int(self.height/2), self.width/self.parameters.scale_factor, self.height/self.parameters.scale_factor], self.particles, self.weights)