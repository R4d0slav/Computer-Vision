from ex2_utils import *

class MeanShiftTracker(Tracker):
    def __init__(self, params):
        self.parameters = params

    def initialize(self, image, region):
        self.x, self.y, self.width, self.height = np.array(region).astype(int)

        # Set width and height to be odd, to match dimentions in creating the histogram
        self.width = self.width + 1 if self.width % 2 == 0 else self.width
        self.height = self.height + 1 if self.height % 2 == 0 else self.height

        patch, _ = get_patch(image, (self.x+int(self.width/2), self.y+int(self.height/2)), (self.width, self.height))
        self.k = create_epanechnik_kernel(patch.shape[0], patch.shape[1], self.parameters.sigma)
        
        self.q = extract_histogram(patch, self.parameters.nbins, weights = self.k)
        self.q = self.q / self.q.sum()
        
        # Set the initial position in the middle of the patch
        self.position = (patch.shape[1]//2, patch.shape[0]//2)

    def track(self, image):
        patch, _ = get_patch(image, (self.x+self.width//2, self.y+self.height//2), (self.width, self.height))
        p = extract_histogram(patch, self.parameters.nbins, weights = self.k)
        p = p / p.sum()
        
        V = np.sqrt(self.q / (p + self.parameters.epsilon))
        bp = backproject_histogram(patch, V, self.parameters.nbins)

        self.position = mean_shift(image = bp, position = self.position, h = self.parameters.h)
        self.x += self.position[0] - int(self.width/2)
        self.y += self.position[1] - int(self.height/2)

        # Update the model
        patch, _ = get_patch(image, (self.x+int(self.width/2), self.y+int(self.height/2)), (self.width, self.height))
        q = extract_histogram(patch, self.parameters.nbins, weights = self.k)
        q = q / q.sum()
        self.q = (1-self.parameters.alpha)*self.q + self.parameters.alpha*q
        self.q = self.q / self.q.sum()
        
        return self.x, self.y, self.width, self.height
        

class MSParams():
    def __init__(self):
        self.nbins = 16
        self.epsilon = 1e-4
        self.h = 25
        self.sigma = 2
        self.alpha = 0.0



def mean_shift(image: np.ndarray, position: tuple, h: int, get_all_positions: bool = False):
    assert h % 2 != 0, 'h must be odd!'
    assert position[0] in range(image.shape[1]) and position[1] in range(image.shape[0]), 'pos must be within image boundaries!'

    n = h // 2
    yi, xi = np.indices((h, h)) - n
    
    I = np.pad(image, n, mode = "constant")
    positions = [(position[0]+n, position[1]+n)]

    while True:
        wi = I[positions[-1][1]-n: positions[-1][1]+n+1, positions[-1][0]-n: positions[-1][0]+n+1]
        
        if np.sum(wi) == 0:
            break
        
        wi = wi / wi.sum()
        shift_vector = np.round(np.sum(wi * (xi, yi), axis=(1, 2)) / wi.sum()).astype(int)
        
        if np.linalg.norm(shift_vector) == 0:
            break

        positions.append(np.clip(positions[-1] + shift_vector, (n, n), (I.shape[1]-1, I.shape[0]-1)))
        if (positions[-2][0] == positions[-1][0]) and (positions[-2][1] == positions[-1][1]):
            break
        
    return [(position[0]-n, position[1]-n) for position in positions] if get_all_positions else [positions[-1][0]-n, positions[-1][1]-n]