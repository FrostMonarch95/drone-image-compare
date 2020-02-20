from __future__ import print_function
import cv2
import numpy as np
import matplotlib.pyplot as plt
from cmp import ret_ssim_img
from cmp import ret_hash_img
from cmp import double_min_max
from cmp import swt_shadow_mask_detect
from cmp import threshold_shadow_mask_detect
from cmp import green_mask_detect
import imutils

MAX_FEATURES = 6000
GOOD_MATCH_PERCENT = 0.8
RATIO=0.8

 
def alignImages(im1, im2):
  print("key point matching")
  # Convert images to grayscale
  im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
  orb=cv2.ORB_create(MAX_FEATURES)
  #kp1,_=orb.detectAndCompute(im1Gray,None) 
  #kp2,_=orb.detectAndCompute(im2Gray,None)
  # Detect freak features and compute descriptors.
  #freak=cv2.xfeatures2d.FREAK_create()
  #freak=cv2.xfeatures2d_FREAK()
  #print("freak create complete")
  # fast = cv2.FastFeatureDetector_create()
  # kp1 = fast.detect(im1Gray,None)
  # kp2=fast.detect(im2Gray,None)
  
  # compute the descriptors with ORB
  # keypoints1, descriptors1 = orb.compute(im1Gray, kp1)
  # keypoints2, descriptors2 = orb.compute(im2Gray,kp2)
  
  
  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray,None) 
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray,None)
  
  
    
  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.NORM_L2)
  matches = matcher.knnMatch(descriptors1, descriptors2, 10)
  good_match=[]
  for idx,ele in enumerate(matches):
      if matches[idx][0].distance <RATIO*matches[idx][1].distance:
          good_match.append(matches[idx][0])
  matches=good_match        
  # Sort matches by score
  matches.sort(key=lambda x: x.distance, reverse=False)
  #print(matches)
  # Remove not so good matches
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]
 
  # Draw top matches
  imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
  cv2.imwrite("matches.jpg", imMatches)
   
  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)
 
  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt
   
  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
 
  # Use homography
  height, width, channels = im2.shape
  im1Reg = cv2.warpPerspective(im1, h, (width, height))
   
  return im1Reg, h

if __name__ == '__main__':
  row,col=3,3
  figure=plt.figure(figsize=(16,8))
  # Read reference image
  refFilename = "2997.jpg"
  print("Reading reference image : ", refFilename)
  imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)
  figure.add_subplot(row,col,1,title="based image")
  plt.imshow(cv2.cvtColor(imReference,cv2.COLOR_BGR2RGB))
  # Read image to be aligned
  imFilename = "3996.jpg"
  print("Reading image to align : ", imFilename);  
  im = cv2.imread(imFilename, cv2.IMREAD_COLOR)
  figure.add_subplot(row,col,2,title="aligned image")
  plt.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
  print("Aligning images ...")
  # Registered image will be resotred in imReg. 
  # The estimated homography will be stored in h. 
  imReg, h = alignImages(im, imReference)
  
  # Write aligned image to disk. 
  outFilename = "aligned.jpg"
  print("Saving aligned image : ", outFilename); 
  cv2.imwrite(outFilename, imReg)

  figure.add_subplot(row,col,3,title="align complete")
  plt.imshow(cv2.cvtColor(imReg,cv2.COLOR_BGR2RGB))
  gray_original=cv2.cvtColor(imReference,cv2.COLOR_BGR2GRAY)
  gray_aligned=cv2.cvtColor(imReg,cv2.COLOR_BGR2GRAY)
  directly_diff=np.abs(np.double(gray_original)-np.double(gray_aligned))
  directly_diff.astype(np.uint8)
  cv2.imwrite("directly_minus.jpg",directly_diff)
  ret2,th2 = cv2.threshold(directly_diff,50,255,cv2.THRESH_BINARY)
  cv2.imwrite("directly_minus_threshold.jpg",th2)
  print("doing minimax_diff")
  
  diff=double_min_max(gray_original,gray_aligned)
  print("minimax_complete")
  print(diff.shape)
  #diff=(diff-np.min(diff))/(np.max(diff)-np.min(diff))*255
  diff=np.uint8(diff)
  cv2.imwrite("double_minni_max.jpg",diff)
  # diff=np.double(gray_aligned)-np.double(gray_original)
  
  # print(diff.shape)
  # diff[diff<0]=0
  # diff=np.uint8(diff)
  ret2,th2 = cv2.threshold(diff,50,255,cv2.THRESH_BINARY)
  minmax_after_threshold=th2.copy()
  cv2.imwrite("double_minni_max_threshold.jpg",th2)
  figure.add_subplot(row,col,4,title='diff')
  plt.imshow(th2,cmap='gray')
  
  ms1=threshold_shadow_mask_detect(gray_original)
  ms2=threshold_shadow_mask_detect(gray_aligned);

  shadow_mask=np.maximum(ms1,ms2)
  cv2.imwrite("shadow_mask.jpg",shadow_mask)

  green_mask=green_mask_detect(imReg)
  cv2.imwrite('green_mask.jpg',green_mask)

  shadow_and_green_mask=np.maximum(green_mask,shadow_mask);
  finmask=np.ones_like(shadow_and_green_mask)
  finmask[shadow_and_green_mask==255]=0
  minmax_after_threshold=minmax_after_threshold*finmask
  minmax_after_threshold=cv2.medianBlur(minmax_after_threshold,3)
  cv2.imwrite("minmax_after_threshold_and_mask.jpg",minmax_after_threshold)
  score,ret_matrix=ret_ssim_img(gray_original,gray_aligned)
  
  print("cmp score ",score)
  figure.add_subplot(row,col,5,title="ssim diff")
  plt.imshow(ret_matrix,cmap='gray')
  print("hash begin")
  _,ret_hash_matrix=ret_ssim_img(gray_original,gray_aligned)
  print("hash end")
  figure.add_subplot(row,col,7,title="hash diff")
  plt.imshow(ret_hash_matrix.astype(np.uint8),cmap='gray')
  if imutils.is_cv3(): 
    (_,cont,__)=cv2.findContours(ret_matrix,2,2)
  else:
    (cont,_)=cv2.findContours(ret_matrix,2,2)
  ssim_diff_show=imReference.copy()
  cv2.drawContours(ssim_diff_show,cont,-1,(0,255,0),4)
  figure.add_subplot(row,col,6,title="ssim contour show")
  plt.imshow(ssim_diff_show)
  
  
  # Print estimated homography
  print("Estimated homography : \n",  h)
  plt.show()