from ex3_utils import *
import importlib
import scipy

try:
    my_module = importlib.import_module('toolkit-dir.utils.tracker')
    Tracker = my_module.Tracker
except ImportError as e:
    pass


class MOSSEparams:
    def __init__(self):
        self.scale_factor = 1.0
        self.sigma = 2
        self.eps = 0.001
        self.alpha = 0.3 # learning rate
        self.use_hanning = True
        self.psr_threshold = 7

    def print(self):
        print("Scale factor:", self.scale_factor)
        print("Sigma:", self.sigma)
        print("Eps:", self.eps)
        print("Alpha:", self.alpha)
        print("Hanning:", self.use_hanning)
        print("PSR thrs:", self.psr_threshold)


class MOSSETracker(Tracker if Tracker else object):
    def __init__(self, parameters = MOSSEparams()):
        self.parameters = parameters

    def name(self):
        return "MOSSE-Tracker"
    
    def calculate_psr(self, correlation_output, max_pos):
        peak = correlation_output.max()
        sidelobe_size = 11

        sidelobe_indices = np.ones_like(correlation_output)
        sidelobe_indices[max_pos[0]-sidelobe_size//2:max_pos[0]+sidelobe_size//2+1, max_pos[1]-sidelobe_size//2:max_pos[1]+sidelobe_size//2+1] = 0
        sidelobe_indices = np.where(sidelobe_indices)

        sidelobe_mean = np.mean(correlation_output[sidelobe_indices])
        sidelobe_std = np.std(correlation_output[sidelobe_indices])

        return (peak-sidelobe_mean) / sidelobe_std

    def preprocess_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

        image_log = np.log(image.astype(np.float32) + 1)

        mean = np.mean(image_log)
        std = np.std(image_log)
        image_norm = (image_log - mean) / std

        return image_norm

    def initialize(self, image, region):
        self.x, self.y, w, h = np.array(region).astype(int)
        
        self.width = int(w * self.parameters.scale_factor)
        self.height = int(h * self.parameters.scale_factor)

        if self.width % 2 == 0:
            self.width += 1

        if self.height % 2 == 0:
            self.height += 1


        F, _ = get_patch(image, (self.x+int(self.width/2), self.y+int(self.height/2)), (self.width, self.height))
        F = self.preprocess_image(F)

        if self.parameters.use_hanning:
            self.hanning = create_cosine_window((self.width, self.height))
            F = F * self.hanning

        F_hat = np.fft.fft2(F)
        F_hat_conj = np.conjugate(F_hat)

        G = create_gauss_peak((self.width, self.height), self.parameters.sigma)
        self.G_hat = np.fft.fft2(G)

        self.N = (self.G_hat * F_hat_conj)
        self.D = (F_hat * F_hat_conj + self.parameters.eps)


    def track(self, image):
        F, _ = get_patch(image, (self.x+self.width//2, self.y+self.height//2), (self.width, self.height))
        F = self.preprocess_image(F)

        if self.parameters.use_hanning:
            F = F * self.hanning

        F_hat = np.fft.fft2(F)
        R = np.fft.ifft2(np.fft.fftshift((self.N / self.D) * F_hat))

        y, x = np.unravel_index(np.argmax(np.abs(R)), R.shape)
        psr = self.calculate_psr(R, (y, x))
    
        if x > self.width / 2:
            x = x - self.width

        if y > self.height / 2:
            y = y - self.height
    
        self.x += x
        self.y += y

        self.x = max(0, min(self.x, image.shape[1]))
        self.y = max(0, min(self.y, image.shape[0]))

        F, _ = get_patch(image, (self.x+int(self.width/2), self.y+int(self.height/2)), (self.width, self.height))
        F = self.preprocess_image(F)

        if self.parameters.use_hanning:
            F = F * self.hanning

        F_hat = np.fft.fft2(F)
        F_hat_conj = np.conjugate(F_hat)

        if psr < self.parameters.psr_threshold: 
            self.N = (self.parameters.alpha*0.5)*(self.G_hat * F_hat_conj) + (1-(self.parameters.alpha*0.5))*self.N
            self.D = (self.parameters.alpha*0.5)*(F_hat * F_hat_conj + self.parameters.eps) + (1-(self.parameters.alpha*0.5))*self.D

        else:
            self.N = self.parameters.alpha*(self.G_hat * F_hat_conj) + (1-self.parameters.alpha)*self.N
            self.D = self.parameters.alpha*(F_hat * F_hat_conj + self.parameters.eps) + (1-self.parameters.alpha)*self.D

        return self.x, self.y, self.width/self.parameters.scale_factor, self.height/self.parameters.scale_factor