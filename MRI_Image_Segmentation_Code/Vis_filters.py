from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

def generate_pattern(model, layer_name, filter_index, size = 256):#!
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:,:,:,filter_index])
    
    grads = K.gradients(loss, model.input)[0] #!
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([model.input], [loss, grads])
    input_img_data = np.random.random((1, size, size, 1)) * 20 + 128 #!
    
    step = 1
    
    for i in range (100):#!
       loss_value, grads_value = iterate([input_img_data])
       input_img_data += grads_value * step
       
    img = input_img_data[0]
    return deprocess_image(img)

def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    
    x += 0.5
    x = np.clip(x, 0, 1)
    
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x
    
def vis_filters(model, num_layers):

    size = 64
    margin = 5
    
    results = np.zeros((8 * size + 7 * margin, 8*size + 7*margin, 1))#!
    for layer in model.layers[1:num_layers]:
        print(layer.name)
        for i in range(8):
            per = (i/8)*100
            print( "%f" %(per))
            for j in range (8):
                filter_img = generate_pattern(model, layer.name, i + (j*8), size = size)
            
                horizontal_start = i *size + i *margin
                horizontal_end = horizontal_start + size
                vertical_start = j * size + j * margin
                vertical_end = vertical_start + size
                results[horizontal_start : horizontal_end, vertical_start: 
                        vertical_end, :] = filter_img
            
        plt.title(layer.name)      
        plt.figure(figsize = (20, 20))
        plt.imshow(results[:,:,0])
            
            
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
