import numpy as np
import cv2

class KCF:

    def __init__(self, frame, rect, padding=1.5, sigma=0.6, lambda_=0.0001, interp_factor=0.012, cell_size=4):
        self.padding = padding
        self.sigma = sigma
        self.lambda_ = lambda_
        self.interp_factor = interp_factor
        self.cell_size = cell_size

        x, y, w, h = tuple(map(int, rect))
        self.pos = np.array([x + w/2, y + h/2], dtype=np.float32)
        self.target_sz = np.array([w, h], dtype=np.int32)
        
        sz = np.floor(self.target_sz * (1 + padding))
        self.sz = (sz // (2 * cell_size) * 2 * cell_size).astype(np.int32)
        self.feat_sz = self.sz // cell_size
        
        self.yf = np.fft.fft2(self._create_gaussian_peak(self.feat_sz, sigma))
        
        self.hann_window = cv2.createHanningWindow((self.feat_sz[0], self.feat_sz[1]), cv2.CV_32F)
        self.hann_window = self.hann_window[:, :, np.newaxis]

        features = self._get_features(frame, self.pos, self.sz)
        self.x_model = features
        
        kf = self._gaussian_correlation(features, features)
        self.alphaf_model = self.yf / (kf + self.lambda_)

    def _create_gaussian_peak(self, size, sigma):
        w, h = size
        xx, yy = np.meshgrid(np.arange(w) - np.floor(w/2), np.arange(h) - np.floor(h/2))
        dist = xx**2 + yy**2
        sy = np.sqrt(w*h) / (2*np.pi) * sigma
        g = np.exp(-0.5 * dist / sy**2)
        g = np.fft.ifftshift(g)
        return g.astype(np.float32)

    def _get_features(self, img: np.ndarray, pos, sz):
        w, h = sz
        xs = np.floor(pos[0]) + np.arange(w) - np.floor(w/2)
        ys = np.floor(pos[1]) + np.arange(h) - np.floor(h/2)
        xs = xs.astype(np.int32)
        ys = ys.astype(np.int32)
        
        left = -min(0, np.min(xs))
        top = -min(0, np.min(ys))
        right = max(0, np.max(xs) - img.shape[1] + 1)
        bottom = max(0, np.max(ys) - img.shape[0] + 1)
        
        if left > 0 or top > 0 or right > 0 or bottom > 0:
            img_padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REPLICATE)
            patch = img_padded[ys[0]+top:ys[-1]+top+1, xs[0]+left:xs[-1]+left+1]
        else:
            patch = img[ys[0]:ys[-1]+1, xs[0]:xs[-1]+1]
            
        if patch.shape[:2] != (h, w):
            patch = cv2.resize(patch, (w, h))

        # HOG
        gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=1)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        
        n_bins = 9
        ang = ang % 180
        bin_width = 180 / n_bins
        
        h_patch, w_patch = patch.shape
        hog_feat = np.zeros((h_patch, w_patch, n_bins), dtype=np.float32)
        
        bin_idx = (ang / bin_width).astype(np.int32) % n_bins
        for i in range(n_bins):
            hog_feat[:, :, i] = mag * (bin_idx == i)
            
        hog_feat_resized = cv2.resize(hog_feat, (self.feat_sz[0], self.feat_sz[1]), interpolation=cv2.INTER_AREA)
        
        if len(hog_feat_resized.shape) == 2:
            hog_feat_resized = hog_feat_resized[:, :, np.newaxis]
            
        # normalization
        hog_feat_resized = (hog_feat_resized - np.mean(hog_feat_resized)) / (np.std(hog_feat_resized) + 1e-5)
        hog_feat_resized = hog_feat_resized * self.hann_window
        
        return hog_feat_resized

    def _gaussian_correlation(self, x1: np.ndarray, x2: np.ndarray):
        x1f = np.fft.fft2(x1, axes=(0, 1))
        x2f = np.fft.fft2(x2, axes=(0, 1))
        
        xyf = np.sum(x1f * np.conj(x2f), axis=2)
        xy = np.real(np.fft.ifft2(xyf))
        
        xx = np.sum(x1**2)
        yy = np.sum(x2**2)
        
        d = (xx + yy - 2 * xy) / (x1.size / x1.shape[2]) 
        d = np.clip(d, 0, None)
        
        k = np.exp(-d / (self.sigma**2))
        return np.fft.fft2(k)

    def _update(self, frame):
        features = self._get_features(frame, self.pos, self.sz)
        
        kf = self._gaussian_correlation(features, self.x_model)
        response_f = self.alphaf_model * kf
        response = np.real(np.fft.ifft2(response_f))
        
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(response)
        
        dx = max_loc[0]
        dy = max_loc[1]
        if dx > self.feat_sz[0] / 2: dx -= self.feat_sz[0]
        if dy > self.feat_sz[1] / 2: dy -= self.feat_sz[1]
        
        self.pos[0] += dx * self.cell_size
        self.pos[1] += dy * self.cell_size
        
        new_features = self._get_features(frame, self.pos, self.sz)
        self.x_model = (1 - self.interp_factor) * self.x_model + self.interp_factor * new_features
        
        kf_new = self._gaussian_correlation(new_features, new_features)
        alphaf_new = self.yf / (kf_new + self.lambda_)
        self.alphaf_model = (1 - self.interp_factor) * self.alphaf_model + self.interp_factor * alphaf_new
        
        x = int(self.pos[0] - self.target_sz[0] / 2)
        y = int(self.pos[1] - self.target_sz[1] / 2)
        return (x, y, self.target_sz[0], self.target_sz[1])