from skimage.measure import compare_ssim
from skimage.morphology import disk
from skimage.filters import rank
import argparse
import cv2
import numpy as np
import imutils
from PIL import Image
import imagehash
import pywt
threshold=10

#hash滑动窗口大小
boxsize=32
#hash 不同时的阈值
hash_threshold=50

def green_mask_detect(bgr_image):
#color_img bgr image
#return green part in color_img
    bgr_image = bgr_image.copy()
    lab_image=cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
    lab_planes=lab_image
    clahe=cv2.createCLAHE(clipLimit=4.0,tileGridSize=(4,4))

    dst=clahe.apply(lab_planes[:,:,0]);

    lab_planes[:,:,0]=dst;

    image_clahe=cv2.cvtColor(lab_planes,cv2.COLOR_LAB2BGR)

    ## convert to hsv
    hsv = cv2.cvtColor(image_clahe, cv2.COLOR_BGR2HSV)


    mask = cv2.inRange(hsv, (20, 10, 20), (90, 255,255))

    mask[mask>0]=255
    return mask

#swt threshold 控制阴影的阈值 越大 阴影区域会检查的越多
swt_threshold=75
def swt_shadow_mask_detect(gray):
#return the shadow mask of gray image;
    level=2
    normalize_img= (np.double(gray)-(np.min(gray)))/(np.max(gray)-np.min(gray))
    retu=pywt.swt2(normalize_img,'db2',level=level)
    mask=retu[1][0]
    cv2.imshow('swt',mask);
    cv2.waitKey(0)
    mask=(mask-np.min(mask))/(np.max(mask)-np.min(mask))*255
    mask=np.uint8(mask)#return back to 0-255
    _,thres_img=cv2.threshold(mask,swt_threshold,255,cv2.THRESH_BINARY_INV)
    return thres_img
def threshold_shadow_mask_detect(gray):
    return cv2.threshold(gray,125,255,cv2.THRESH_BINARY_INV)[1];




def crop_black_border(grayA,grayB):
#gray A: the image should be cut with the gray B black border size
#gray B: the image with black border
    img=grayB.copy()
    img_cut=grayA.copy()

    _,img=cv2.threshold(img,1,255,cv2.THRESH_BINARY_INV)
    if imutils.is_cv3():
        (_,cont,__)=cv2.findContours(img,2,2)
    else:
        (cont,_)=cv2.findContours(img,2,2)
    
    mask=np.zeros(img.shape,dtype=np.uint8)
    cv2.drawContours(mask,cont,-1,255,-1)
 
    
    img_cut[mask==255]=0
    return img_cut
#input two gray scale images return cmp score and result matrix
#gray B should be the matrix being warpped we assume there is black edge in it!
def ret_ssim_img(grayA,grayB):
    # 5. Compute the Structural Similarity Index (SSIM) between the two
    #    images, ensuring that the difference image is returned
    grayA=crop_black_border(grayA,grayB)
    (score, diff) = compare_ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    ret=np.zeros(diff.shape,dtype=np.uint8)
    ret[diff<threshold]=255
    return score,ret
#基于hash的图像比较
#图像经过行列滑窗得到32*32的子窗口，然后比较hash值，hash超过一定范围，我们认为区域不同
def ret_hash_img(grayA,grayB):
    #print("hash image")
    grayA=crop_black_border(grayA,grayB)
    grayA=Image.fromarray(grayA)
    grayB=Image.fromarray(grayB)
    c,r=grayA.size
    mask=np.zeros((r,c),dtype=np.uint8)
    for i in range(r-boxsize+1):
        row=i*boxsize
        for j in range(c-boxsize+1):
            col=j*boxsize
            right=col+boxsize-1
            down=row+boxsize-1
            grayA_sub=grayA.crop((col,row,right,down))
            grayB_sub=grayB.crop((col,row,right,down))
            hash1=imagehash.average_hash(grayA_sub)
            hash2=imagehash.average_hash(grayB_sub)
            dif=hash1-hash2
            if dif>hash_threshold:mask[row:down+1,col:right+1]=255
    return mask    
#输入两张单通道 灰度图
#必须保证尺寸一致
#3*3 sliding window for histogram matching and double min max
def double_min_max(grayA,grayB):
    grayA=crop_black_border(grayA,grayB)
    
    assert(grayA.shape==grayB.shape)
    assert(len(grayA.shape)==2)
    grayA=hist_match(grayA,grayB)
    

    r,c=grayA.shape
    grayA=grayA.astype(np.int16)
    grayB=grayB.astype(np.int16)
    img_group=[grayA,grayB]
    
    nimg=img_group[0]
    idx=0
    out=np.zeros(grayA.shape,dtype=np.int16)
    for count,x in enumerate(np.nditer(nimg)):
        nr=count//c
        nc=count%c
        if not nr or nr == r-1:continue
        
        if not nc or nc==c-1:continue
        
        
        # tmpls=[]
        minval= np.min(np.abs(np.ones((3,3)).astype(np.int16)* x-img_group[1-idx][nr-1:nr+2,nc-1:nc+2]))
        
        out[nr,nc]=minval
        # print(tmpls)
    
    # ans=np.maximum(ls[0],ls[1])
    return out
import numpy as np

def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)
    
        
    
    
    