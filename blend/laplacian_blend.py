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

def downsample(image:np.ndarray, kernel:np.ndarray) -> np.ndarray:

    downimage = np.copy(image)
    for channel in range(3):
        sc = downimage[:,:,channel]
        sc = conv(sc, kernel)
        downimage[:,:,channel] = sc
    
    return downimage[::2, ::2]

def upsample(image:np.ndarray, kernel:np.ndarray) -> np.ndarray:
    
    height, width = image.shape[:2]
    up_height = height * 2
    up_width = width * 2

    upimage = np.zeros((up_height, up_width, 3), dtype=np.float32)

    for channel in range(3):

        scbg = np.zeros((up_height, up_width), dtype=np.float32)
        scbg[::2, ::2] = image[:,:,channel]
        
        sc = conv(scbg, kernel*4)
        upimage[:,:,channel] = sc

    return upimage

def gaussian_pyramid(img:np.ndarray, kernel:np.ndarray, levels) -> list:
    
    img = img.astype(np.float32)
    gplist = [img]
    
    for i in range(levels):
        print(f"Gauss Pyramid Level {i}")
        g = downsample(gplist[i], kernel)
        gplist.append(g)

    return gplist

def laplacian_pyramid(gplist:list[np.ndarray], kernel:np.ndarray) -> list:

    levels=len(gplist)
    lplist=[]

    for i in range(levels-1):
        print(f"Laplacian Pyramid Level {i}")
        g_expanded = upsample(gplist[i+1], kernel)
        
        h, w, _ = gplist[i].shape
        g_expanded = g_expanded[:h, :w, :]
        
        lplist.append(gplist[i] - g_expanded)

    lplist.append(gplist[-1])

    return lplist

def reconstruct(lplist:list[np.ndarray], kernel:np.ndarray) -> np.ndarray:

    levels=len(lplist)
    image=lplist[-1]

    for i in range(levels-2,-1,-1):
        print(f"Reconstruct Level {i}")
        upimage = upsample(image, kernel)
        
        h, w, _ = lplist[i].shape
        upimage = upimage[:h, :w, :]
        
        image = upimage + lplist[i]

    return image

def plot_plist(type, plist:list[np.ndarray], ypos, totallines):

    levels=len(plist)
    for i in range(levels):
        plt.subplot(totallines, levels, (i+1) + ypos*levels)
        img=plist[i]

        disp=img.copy().astype(np.float32)

        if type=='L' and i<levels-1:
            disp = 4*abs(disp)
        
        disp=np.clip(disp, 0, 255)
        plt.imshow(disp.astype(np.uint8))
        plt.title(f'{type}{i}')
        plt.axis('off')


def blend_img(imga:np.ndarray, imgb:np.ndarray, mask:np.ndarray) -> np.ndarray:
    
    blended = imga.astype(np.float32)
    blended = imga*mask + imgb*(1-mask)
    return blended.astype(np.float32)


if __name__ == '__main__':

    # apple orange 

    # img1=Image.open('apple.png')
    # img2=Image.open('orange.png')

    # img1=np.array(img1)
    # img2=np.array(img2)

    # mask=np.zeros_like(img1).astype(np.float32)
    # mask[:, :mask.shape[1]//2, :] = 1.0

    # real image & mask

    img1=Image.open('reala.jpg')
    img2=Image.open('realb.jpg')
    mask=Image.open('realmask.jpg')

    img1=np.array(img1)
    img2=np.array(img2)
    mask=np.array(mask)/255.0

    levels=13
    kernel_size=27
    kernel_sigma=0.15*kernel_size+0.35
        
    kernel=gausskernel(kernel_size, kernel_sigma)
    
    print("img1")
    img1_g=gaussian_pyramid(img1, kernel, levels)
    img1_l=laplacian_pyramid(img1_g, kernel)

    print("img2")
    img2_g=gaussian_pyramid(img2, kernel, levels)
    img2_l=laplacian_pyramid(img2_g, kernel)

    print("mask")
    mask_g=gaussian_pyramid(mask, kernel, levels)
    # plot_plist('mask',mask_g,0,1)


    totallines=4
    plt.figure(figsize=(30, 40))



    fused_l=img1_l.copy()
    for i in range(levels):
        fused_l[i]=blend_img(img1_l[i], img2_l[i], mask_g[i])

    # plot_plist('G',img1_g,0,totallines)
    # plot_plist('L',img1_l,1,totallines)
    # plot_plist('G',img2_g,2,totallines)
    # plot_plist('L',img2_l,3,totallines)
    # plt.show()

    fused_image=reconstruct(fused_l, kernel)
    fused_image=np.clip(fused_image, 0, 255)

    plt.imshow(fused_image.astype(np.uint8))
    plt.axis('off')
    plt.savefig('real_fuse_uplevel.png')
    plt.show()