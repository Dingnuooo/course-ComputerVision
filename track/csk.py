import numpy as np
import cv2

class CSK:
    
    def __init__(self, frame, rect, padding=1.0, sigma=0.2, lambda_=0.01, interp_factor=0.02):
        self.padding = padding
        self.sigma = sigma
        self.lambda_ = lambda_
        self.interp_factor = interp_factor

        x, y, w, h = tuple(map(int, rect))
        self.pos = np.array([x + w/2, y + h/2], dtype=np.float32)
        self.target_sz = np.array([w, h], dtype=np.int32)
        
        self.sz = np.floor(self.target_sz * (1 + padding)).astype(np.int32)
        
        self.y = self._create_gaussian_peak(self.sz, sigma)
        self.yf = np.fft.fft2(self.y)

        self.hann_window = cv2.createHanningWindow((self.sz[0], self.sz[1]), cv2.CV_32F)

        patch = self._get_subwindow(frame, self.pos, self.sz)
        self.x_model = patch
        
        kf = self._dense_gauss_kernel(self.x_model, self.x_model)
        self.alphaf_model = self.yf / (kf + self.lambda_)

    def _create_gaussian_peak(self, size, sigma):
        w, h = size
        xx, yy = np.meshgrid(np.arange(w) - np.floor(w/2), np.arange(h) - np.floor(h/2))
        dist = xx**2 + yy**2
        sy = np.sqrt(w*h) / (2*np.pi) * sigma
        g = np.exp(-0.5 * dist / sy**2)
        g = np.fft.ifftshift(g)
        return g.astype(np.float32)

    def _get_subwindow(self, img: np.ndarray, pos, sz):
        w, h = sz
        xs = np.floor(pos[0]) + np.arange(w) - np.floor(w/2)
        ys = np.floor(pos[1]) + np.arange(h) - np.floor(h/2)
        
        xs = xs.astype(np.int32)
        ys = ys.astype(np.int32)
        
        # padding border
        left = -min(0, np.min(xs))
        top = -min(0, np.min(ys))
        right = max(0, np.max(xs) - img.shape[1] + 1)
        bottom = max(0, np.max(ys) - img.shape[0] + 1)
        
        if left > 0 or top > 0 or right > 0 or bottom > 0:
            img_padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REPLICATE)
            patch = img_padded[ys[0]+top:ys[-1]+top+1, xs[0]+left:xs[-1]+left+1]
        else:
            patch = img[ys[0]:ys[-1]+1, xs[0]:xs[-1]+1]
            
        # 确保尺寸完全匹配（防止舍入误差）
        if patch.shape != (h, w):
            patch = cv2.resize(patch, (w, h))

        patch = (patch - np.mean(patch)) / (np.std(patch) + 1e-5)
        patch = patch * self.hann_window
        return patch

    def _dense_gauss_kernel(self, x1, x2):
        x1f = np.fft.fft2(x1)
        x2f = np.fft.fft2(x2)
        c = np.real(np.fft.ifft2(x1f * np.conj(x2f)))
        d = np.sum(x1**2) + np.sum(x2**2) - 2 * c
        k = np.exp(-1 / (self.sigma**2) * np.clip(d, 0, None) / np.size(x1))
        return np.fft.fft2(k)

    def _update(self, frame):
        patch = self._get_subwindow(frame, self.pos, self.sz)
        kf = self._dense_gauss_kernel(patch, self.x_model)
        response_f = self.alphaf_model * kf
        response = np.real(np.fft.ifft2(response_f))
        
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(response)
        
        dx = max_loc[0]
        dy = max_loc[1]
        if dx > self.sz[0] / 2: dx -= self.sz[0]
        if dy > self.sz[1] / 2: dy -= self.sz[1]
        
        self.pos[0] += dx
        self.pos[1] += dy
        
        new_patch = self._get_subwindow(frame, self.pos, self.sz)
        self.x_model = (1 - self.interp_factor) * self.x_model + self.interp_factor * new_patch
        
        kf_new = self._dense_gauss_kernel(new_patch, new_patch)
        alphaf_new = self.yf / (kf_new + self.lambda_)
        self.alphaf_model = (1 - self.interp_factor) * self.alphaf_model + self.interp_factor * alphaf_new
        
        x = int(self.pos[0] - self.target_sz[0] / 2)
        y = int(self.pos[1] - self.target_sz[1] / 2)
        return (x, y, self.target_sz[0], self.target_sz[1])