import skimage.io as io
import numpy as np
import matplotlib.pylab as plt

def limit(mask, img, dicom_file):
    for i in range(255):
        for j in range(255):
            if((not (mask[i,j] == mask[i,j+1] == mask[i,j-1] == mask[i-1, j] == 
               mask[i-1, j-1] == mask[i-1, j+1] == mask[i+1, j] == 
               mask[i+1, j-1] == mask[i+1,j+1])) and (mask[i,j] != 0)):
                if(dicom_file == False):
                    img[i,j]=255
                else:
                    img[i,j]=1
                
    return img

def show_organ(mask, img):
    for i in range(256):
        for j in range (256):
            if mask[i,j] == 0:
                img[i,j] = 0
    
    return img
            
            
'''
def show(imgs, size):
    images_per_row = 16
    

    for img in imgs:
        n_cols = 1 
        images_per_row = 8
        display_grid = np.zeros((size * n_cols, images_per_row * size))


        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = imgs[0,:, :,0]
                Post-process the feature to make it visually palatable
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                
                display_grid[col * size : (col + 1) * size,
                            row * size : (row + 1) * size] = channel_image
              
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(col)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
    
    plt.show()
'''              
                
    
               
        
                
    
