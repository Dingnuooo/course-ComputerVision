import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from matplotlib import use as muse
muse('TkAgg')

def conv(image:np.ndarray, kernel:np.ndarray) -> np.ndarray:
    # single channel image
    kh, kw = kernel.shape
    padding = np.pad(image, ((kh//2,kh//2), (kw//2,kw//2)), 'reflect')
    result = np.zeros_like(image, dtype=np.float32)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padding[i:i+kh, j:j+kw]
            result[i, j] = np.sum(region*kernel)

    return result

def gausskernel(size, sigma=1.0) -> np.ndarray:

    ax = np.linspace(-(size//2), size//2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2+yy**2) / (2.0*sigma**2))
    return kernel / np.sum(kernel)

def plot_mask(mask:np.ndarray):

    mask_disp = mask*255
    plt.imshow(mask_disp.astype(np.uint8))

def blend_img(imga:np.ndarray, imgb:np.ndarray, mask:np.ndarray) -> np.ndarray:
    
    blended = imga.astype(np.float32)
    blended = imga*mask + imgb*(1-mask)
    return blended.astype(np.float32)

if __name__ == '__main__':

    img1=Image.open('apple.png')
    img2=Image.open('orange.png')

    img1=np.array(img1)
    img2=np.array(img2)

    mask=np.zeros_like(img1[:,:,1]).astype(np.float32)
    mask[:, :mask.shape[1]//2] = 1.0

    blendlist=[]
    
    plt.figure(figsize=(12, 8))

    # # pingjie
    mask=np.stack([mask,mask,mask], axis=2)
    blended=blend_img(img1, img2, mask)
    plt.imshow(blended.astype(np.uint8))
    plt.xticks(np.arange(0,512,128))
    plt.yticks(np.arange(0,512,128))
    plt.show()

    sizelist = [5,9,27,81,243,729]

    for kernel_size in sizelist:

        kernel_sigma=kernel_size/6
        
        kernel=gausskernel(kernel_size, kernel_sigma)
        softmask_sc=conv(mask, kernel)
        
        softmask=np.stack([softmask_sc, softmask_sc, softmask_sc], axis=2)

        blended=blend_img(img1, img2, softmask)
        blendlist.append(blended)

    for i in range(0,6):
        plt.subplot(2,3,i+1)
        plt.title(f'kernel size={sizelist[i]}')
        plt.xticks(np.arange(0,512,128))
        plt.yticks(np.arange(0,512,128))
        plt.imshow(blendlist[i].astype(np.uint8))

    plt.show()