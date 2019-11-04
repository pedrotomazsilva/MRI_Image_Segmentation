from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import tensorflow as tf
from model_2 import dice_coef
from show_results import limit, show_organ
import pydicom as dicom


#Ajusta os dados da imagem e da máscara para valores entre 0 e 1

def adjustData(img,mask):   

    if(np.max(img) <= 1.0):
        img = img
        #print("Imagem com codificação entre 0 e 1") 
            
    else:        
        #print("Imagem com codificação entre 0 e 255") 
        img = (img-np.min(img))/(np.max(img)-np.min(img)) #Transformação linear que coloca os valores entre 0 e 1
        
    if(np.max(mask) <= 1.0):
        mask = mask
        #print("Máscara com codificação entre 0 e 1") 
            
    else:
        #print("Máscara com codificação entre 0 e 255") 
        mask = (mask-np.min(mask))/(np.max(mask)-np.min(mask)) #Transformação linear que coloca os valores entre 0 e 1
    
  
    return (img,mask)




#Cria as imagens e máscaras de treino para o modelo realizando data augmentation
#Para garantir a mesma transformação na imagem e máscara seed deve ter o mesmo valor

def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,
                   image_color_mode = "grayscale",
                   mask_color_mode = "grayscale",image_save_prefix  = "image",
                   mask_save_prefix  = "mask"
                   ,save_to_dir = "aug/endocardio",
                   target_size = (256,256),seed = 1):
      
    #colocar **aug_dict no argumento de image_datagen e mask_satagen para data augmentation
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    
    
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    
    
    train_generator = zip(image_generator, mask_generator)
    

    for (img,mask) in train_generator:
        
        img,mask = adjustData(img,mask)
      
        yield (img,mask)




#Cria as imagens de teste para o modelo
        
def testGenerator(test_path, num_image, dicom_file, target_size = (256,256),as_gray = True):
   
    for i in range(num_image):
        
        dirs = os.listdir(test_path)
        
        if(dicom_file == False):
            img = io.imread(os.path.join(test_path,"%d.png"%i),
                            as_gray = as_gray).astype(np.float32)
            
            img = img - np.mean(img)
            img = img/np.std(img)
        
            if(np.max(img) <= 1.0):
                print("Imagem de teste com codificação entre 0 e 1") 
            
            else:        
                img = (img-np.min(img))/(np.max(img)-np.min(img)) #Transformação linear que coloca os valores entre 0 e 1

        else:
            img = dicom.dcmread(os.path.join(test_path,dirs[i]))
            img = img.pixel_array
            img = img - np.mean(img)
            img = img/np.std(img)
            img = (img-np.min(img))/(np.max(img)-np.min(img)) #Transformação linear que coloca os valores entre 0 e 1
       
        
    
        img = trans.resize(img,target_size)
        
        img = np.reshape(img,img.shape+(1,)) 
    
        img = np.reshape(img,(1,)+img.shape)
    
        
        yield img




#Cria as imagens e máscaras de validação para o modelo
        
def validationGenerator(batch_size,validation_path,image_folder,mask_folder,
                        aug_dict,image_color_mode = "grayscale",
                        mask_color_mode = "grayscale",
                        image_save_prefix  = "image",
                        mask_save_prefix  = "mask",
                        flag_multi_class = False,target_size = (256,256),
                        seed = 1, 
                        save_to_dir_imageval = "aug/validation/brain/image",
                        save_to_dir_labelval = "aug/validation/brain/label"):
    
   
    image_datagen_validation = ImageDataGenerator(samplewise_std_normalization=True,
                                                  samplewise_center=True
                                                  )
    mask_datagen_validation = ImageDataGenerator(samplewise_std_normalization=True,
                                                 samplewise_center=True
                                                 )
    
    image_generator_validation = image_datagen_validation.flow_from_directory(
        validation_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir_imageval,
        save_prefix  = image_save_prefix,
        seed = seed)
    
       
    mask_generator_validation = mask_datagen_validation.flow_from_directory(
        validation_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir_labelval,
        save_prefix  = mask_save_prefix,
        seed = seed)
    
    
    validation_generator = zip(image_generator_validation, 
                               mask_generator_validation)
    

    for (img_val,mask_val) in validation_generator:
        
        img_val,mask_val = adjustData(img_val,mask_val)
        
        yield (img_val,mask_val)

    
   

#Guarda os resultados para save_path
        
def saveResult(save_path, test_path, npyfile, dicom_file):
    
    dirs = os.listdir(test_path)

    for i,item in enumerate(npyfile):
        if(dicom_file == False):
            img = io.imread(os.path.join(test_path, "%d.png"%i), 
                            as_gray = True).astype(np.uint8)
        else:
            img = dicom.dcmread(os.path.join(test_path,dirs[i]))
            img = img.pixel_array
            img = (img-np.min(img))/(np.max(img)-np.min(img)) #Normaliza valores
        

        mask = item[:,:,0]
        
        mean = np.mean(mask) #coloca os valores da máscara exclusivamente a 0 ou 1
        mask[mask>=mean]=1
        mask[mask<mean]=0
        
        img = trans.resize(img,(256,256))
        
        #np.set_printoptions(threshold=np.nan)
        np.save(os.path.join(save_path,"%d_predict_npy"%i),mask)
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),mask)
        limited = limit(mask, img, True)
        print("Image %d"%i)
        io.imsave(os.path.join(save_path,"%d_predict_contour.png"%i),limited)
        #io.imsave(os.path.join(save_path,"%d_predict_image.png"%i),img)
        #organ = show_organ(mask, img) para mostrar apenas o orgão desativar a função limit
        #io.imsave(os.path.join(save_path,"%d_predict_organ.png"%i),organ)
          
def evaluateResults(predicts_path, test_path, num_imgs):
        
    print(num_imgs)
    avg_val_coef = 0
    i=0
    for i in range(num_imgs):
        img_predict = np.load(os.path.join(predicts_path,"%d_predict_npy.npy"%i))
        
        if(np.max(img_predict) > 1.0):
            print("imagem nao esta entre 0 e 1")
            img_predict = (img_predict-np.min(img_predict))/(np.max(img_predict)-np.min(img_predict)) #Transformação linear que coloca os valores entre 0 e 1
            
  
        img_test_label = io.imread(os.path.join(test_path,"%d.png"%i), 
                                   as_gray = True).astype(np.float32)
       
        
        if(np.max(img_test_label) > 1.0):
            img_test_label = (img_test_label-np.min(img_test_label))/(np.max(img_test_label)-np.min(img_test_label))
            
        print("%d"%i)
        val_coef = tf.keras.backend.eval(dice_coef(img_test_label, img_predict))
        print(val_coef)
        avg_val_coef = avg_val_coef + val_coef
        
    avg_val_coef = avg_val_coef/num_imgs
    print("Média do dice_coef")
    print(avg_val_coef)
   
    
    