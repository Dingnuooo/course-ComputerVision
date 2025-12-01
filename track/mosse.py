import numpy as np
import cv2

class MOSSE:
    def __init__(self, frame, rect, learning_rate=0.125, sigma=2.0, num_train=128):
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.num_train = num_train

        x, y, w, h = tuple(map(int, rect))
        self.pos = (x + w/2, y + h/2)
        self.size = (w, h)
        
        self.hanning_window = cv2.createHanningWindow((w, h), cv2.CV_32F)
        self.G = self._gaussian_response(w, h)

        self.H, self.A, self.B = self._init_training(frame)

    def _gaussian_response(self, w, h):
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        center_x = w // 2
        center_y = h // 2
        dist = (xx - center_x)**2 + (yy - center_y)**2
        g = np.exp(-dist / (2 * self.sigma**2))
        g = (g - g.min()) / (g.max() - g.min() + 1e-5)
        G = np.fft.fft2(g)
        return G

    def _pre_process(self, img):
        img = np.log(img + 1.0)
        img = (img - np.mean(img)) / (np.std(img) + 1e-5)
        img = img * self.hanning_window
        return img

    def _rand_warp(self, img:np.ndarray):
        h, w = img.shape[:2]
        C = 0.1
        ang = np.random.uniform(-C, C)
        c, s = np.cos(ang), np.sin(ang)
        M = np.array([[c, -s, 0], [s, c, 0]])
        center_warp = np.array([[w/2], [h/2]])
        tmp = np.dot(M, np.vstack([center_warp, [1]]))
        M[:, 2] = center_warp.flatten() - tmp.flatten() + \
                  np.array([np.random.uniform(-C, C)*w, np.random.uniform(-C, C)*h])
        warped = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return warped

    def _get_rect_sub_pix(self, frame: np.ndarray, center, size):
        w, h = size
        cx, cy = center
        img_patch: np.ndarray = cv2.getRectSubPix(frame, (w, h), (cx, cy))
        if img_patch is None or img_patch.shape != (h, w):
            x1 = int(cx - w/2)
            y1 = int(cy - h/2)
            x2 = x1 + w
            y2 = y1 + h
            img_h, img_w = frame.shape
            xs1 = max(0, x1); xs2 = min(img_w, x2)
            ys1 = max(0, y1); ys2 = min(img_h, y2)
            if xs2 <= xs1 or ys2 <= ys1:
                return np.zeros((h, w), dtype=np.float32)
            img_patch = np.zeros((h, w), dtype=np.float32)
            img_patch[ys1-y1:ys2-y1, xs1-x1:xs2-x1] = frame[ys1:ys2, xs1:xs2]
        return img_patch.astype(np.float32)

    def _init_training(self, frame):
        img_roi = self._get_rect_sub_pix(frame, self.pos, self.size)
        A = np.zeros_like(self.G)
        B = np.zeros_like(self.G)
        
        F = np.fft.fft2(self._pre_process(img_roi))
        A += self.G * np.conj(F)
        B += F * np.conj(F)

        for _ in range(self.num_train):
            warped_img = self._rand_warp(img_roi)
            F = np.fft.fft2(self._pre_process(warped_img))
            A += self.G * np.conj(F)
            B += F * np.conj(F)
            
        H = A / (B + 1e-5)
        return H, A, B

    def _update(self, frame):
        w, h = self.size

        img_roi = self._get_rect_sub_pix(frame, self.pos, self.size)
        F = np.fft.fft2(self._pre_process(img_roi))

        R_freq = F * self.H
        response = np.fft.ifft2(R_freq)
        response = np.real(response)
        
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(response)
        
        dx = max_loc[0] - w // 2
        dy = max_loc[1] - h // 2
        
        self.pos = (self.pos[0] + dx, self.pos[1] + dy)
        
        new_roi = self._get_rect_sub_pix(frame, self.pos, self.size)
        F_new = np.fft.fft2(self._pre_process(new_roi))
        
        self.A = self.learning_rate * (self.G * np.conj(F_new)) + (1 - self.learning_rate) * self.A
        self.B = self.learning_rate * (F_new * np.conj(F_new)) + (1 - self.learning_rate) * self.B
        self.H = self.A / (self.B + 1e-5)
            
        x = int(self.pos[0] - w / 2)
        y = int(self.pos[1] - h / 2)
        return (x, y, w, h)