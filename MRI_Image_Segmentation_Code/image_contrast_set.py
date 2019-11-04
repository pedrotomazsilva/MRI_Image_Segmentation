from skimage import exposure
import skimage.io as io
import os
import numpy as np
import matplotlib.pyplot as plot

for i in range(3000):   
    img = io.imread(os.path.join("data/test/endocardio/image_original","%d.png"%i), as_gray = True)#.astype(np.float32)
    #img = exposure.equalize_hist(img)
    #img = exposure.equalize_adapthist(img, clip_limit=0.03)
    p2, p98 = np.percentile(img, (2, 98))
    img = exposure.rescale_intensity(img, in_range=(p2, p98))
    io.imsave(os.path.join("aug/endocardio","%d.png"%i), img)
                 
