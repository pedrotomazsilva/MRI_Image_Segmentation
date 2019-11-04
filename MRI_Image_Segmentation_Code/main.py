from model_4 import unet
from data import trainGenerator, validationGenerator, testGenerator, saveResult, evaluateResults
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import os
from Vis_layers import vis_layers
from Vis_filters import vis_filters


#Para visualizar gráficos colocar o seguinte comando no cmd: 
#tensorboard --logdir=logs
#Deve-se abrir o navegador de internet no endereço indicado no cmd
tensorboard = TensorBoard(log_dir='logs/run_n', write_graph=True, 
                          write_images=True)


num_imgs_validation = len(os.listdir("data/validation/epicardio/image"))
num_imgs_train = len(os.listdir("data/train/epicardio/image"))
num_imgs_test = len(os.listdir("data/test/epicardio/image"))
fator_img_aug = 10 #com data augmentation este valor é 10
batch_size = 1


#dicionário que contém os parâmetros para data augmentation
data_gen_args = dict(samplewise_std_normalization=True,
                    samplewise_center=True,
                    rotation_range=25,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=5,
                    zoom_range=0.5,
                    horizontal_flip=True,
                    fill_mode='nearest')



#Criam-se os dados de treino e de validação
#Para gravar estes dados definir uma localização para as variáveis save_to_dir,
#save_to_dir_imageval e save_to_dir_labelval                              
trainGene = trainGenerator(batch_size,'data/train/epicardio','image','label',
                           data_gen_args,
                           save_to_dir = None)
validationGene = validationGenerator(batch_size,'data/validation/epicardio',
                                     'image','label', 
                                     data_gen_args,
                                     save_to_dir_imageval = None, 
                                     save_to_dir_labelval = None)



#Keras callbacks
model_checkpoint = ModelCheckpoint('unet_epicardio.hdf5', monitor='val_loss',
                                   verbose=1, save_best_only=True)
#monitor_lr = ReduceLROnPlateau(monitor='loss', factor = 0.3, patience = 5)

#Estabelecem-se os modelos
#model_vis = unet(1) #modelo para visualizações
#model_vis.load_weights('unet_epicardio.hdf5')
model = unet(0)
model.load_weights('unet_epicardio.hdf5')


#Visualização das camadas e dos filtros da rede
#vis_layers("data/test/epicardio/image/2.png", (256,256), model_vis, num_layers = 35)
#vis_filters(model, num_layers = 35)

#Treina-se o modelo
#Caso se queira treinar o modelo de novo deve anular-se model.load_weights
model.fit_generator(trainGene,epochs=100, steps_per_epoch = 
                    (num_imgs_train*fator_img_aug/batch_size), 
                    validation_data = validationGene, 
                    validation_steps = 
                    (num_imgs_validation*fator_img_aug)/batch_size, 
                    callbacks=[model_checkpoint, tensorboard])






#Aplica-se o modelo treinado às imagens de teste e procede-se à gravação dos 
#resultados
testGene = testGenerator("data/test/epicardio/image", num_image = num_imgs_test, dicom_file = False)
validationGene = testGenerator("data/validation/epicardio/image", num_image = num_imgs_validation, dicom_file = False)


print("----Creating and saving test prediction results----")
results_test = model.predict_generator(testGene, num_imgs_test, verbose=1)
saveResult("predicts/epicardio/test", "data/test/epicardio/image", results_test, dicom_file = False)
'''
print("----Creating and saving validation prediction results----")
results_validation = model.predict_generator(validationGene, num_imgs_validation, verbose=1)
saveResult("predicts/epicardio/validation", "data/validation/epicardio/image", results_validation, dicom_file = False)
'''
#Avaliação dos resultados
evaluateResults(predicts_path = "predicts/epicardio/test", 
                test_path = "data/test/epicardio/label", num_imgs = num_imgs_test)

