from ex3_utils import *
import importlib

try:
    my_module = importlib.import_module('toolkit-dir.utils.tracker')
    Tracker = my_module.Tracker
except ImportError as e:
    pass


class CFparams:
    def __init__(self):
        self.scale_factor = 1.1
        self.sigma = 2
        self.lmbd = 0.1
        self.alpha = 0.2 # learning rate
        self.use_hanning = True

    def print(self):
        print("Scale factor:", self.scale_factor)
        print("Sigma:", self.sigma)
        print("Lmbd:", self.lmbd)
        print("Alpha:", self.alpha)
        print("Hanning:", self.use_hanning)


class CFTracker(Tracker if Tracker else object):
    def __init__(self, parameters = CFparams()):
        self.parameters = parameters

    def name(self):
        return "CF-Tracker"
    
    def preprocess_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

        image_log = np.log(image + 1)

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

        self.H_hat_conj = (self.G_hat * F_hat_conj) / (F_hat * F_hat_conj + self.parameters.lmbd)


    def track(self, image):
        F, _ = get_patch(image, (self.x+self.width//2, self.y+self.height//2), (self.width, self.height))
        F = self.preprocess_image(F)

        if self.parameters.use_hanning:
            F = F * self.hanning

        F_hat = np.fft.fft2(F)
        R = np.fft.ifft2(self.H_hat_conj * F_hat)
        y, x = np.unravel_index(np.argmax(R), R.shape)
    
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
        H_hat_conj2 = (self.G_hat * F_hat_conj) / (F_hat * F_hat_conj + self.parameters.lmbd)
        self.H_hat_conj = (1-self.parameters.alpha) * self.H_hat_conj + self.parameters.alpha * H_hat_conj2

        return self.x, self.y, self.width/self.parameters.scale_factor, self.height/self.parameters.scale_factor

