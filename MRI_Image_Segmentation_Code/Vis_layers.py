from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt


def vis_layers(img_path, target_size, model, num_layers):
    
    img = image.load_img(img_path, target_size)
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255
    
   
    
    activations = model.predict(img_tensor)
    
    
   
    layer_names = []
    for layer in model.layers[1:num_layers]:
        layer_names.append(layer.name)
    

    images_per_row = 16
    

    for layer_name, layer_activation in zip(layer_names, activations):   
    
        n_features = layer_activation.shape[-1]


        size = layer_activation.shape[1]

    
        if(images_per_row <= n_features):
            n_cols = n_features // images_per_row  
        else: 
            n_cols = 1 
            images_per_row = n_features
        display_grid = np.zeros((size * n_cols, images_per_row * size))

   
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0,
                                                 :, :,
                                                 col * images_per_row + row]
    
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
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
    
    plt.show()
    