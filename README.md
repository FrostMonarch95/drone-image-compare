# drone-image-compare
Drone image compare including sift registration,shadow detection, plant detection and a optimized image difference


Firstly, we load 2 images and begin to do registration

First day
![Image description](https://github.com/FrostMonarch95/drone-image-compare/blob/master/2997.jpg)

Second day
![Image description](https://github.com/FrostMonarch95/drone-image-compare/blob/master/3996.jpg)

Registration(the second day image does warpaffine)
![Image description](https://github.com/FrostMonarch95/drone-image-compare/blob/master/aligned.jpg)
****
Then we detect shadow area(they are low confidence area so we decide to give it up.An efficent and real time shadow removal is quite hard for me)

![Image description](https://github.com/FrostMonarch95/drone-image-compare/blob/master/shadow_mask.jpg)

Also we are not intersted in green plants so we also detect green plants in an image

![Image description](https://github.com/FrostMonarch95/drone-image-compare/blob/master/green_mask.jpg)
****
finally we have the changing area with the removal of the above masks. I will really appreciate someone can tell me how to remove those roads, because they are not changed at all.

![Image description](https://github.com/FrostMonarch95/drone-image-compare/blob/master/minmax_after_threshold_and_mask.jpg)
