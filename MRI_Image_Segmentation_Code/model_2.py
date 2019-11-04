from keras.models import Model
from keras.layers import Input, concatenate, SeparableConv2D, MaxPooling2D, Conv2DTranspose, Dropout 
from keras.optimizers import Adam
from keras import backend as K



smooth = 1

#Cálculo do Coeficiente de Dice para uma máscara prevista pelo modelo

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
   
    return (2. * intersection + smooth) / (K.sum(y_true_f) + 
            K.sum(y_pred_f) + smooth)




#Valor do coeficiente de Dice a passar para a avaliação do modelo
#Como se pretende minimizar a função de custo (loss) deve tomar-se o simétrico
#do coeficiente de Dice
    
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)



#Rede convolucional unet, com:
#dropout na quarta e quinta camadas para reduzir o nível de overfit
#otimizador - Adam
#função de custo (loss_function) - Coeficiente de Dice
#Inicializador de pesos (weights ou kernel) - he_normal

def unet(vis):
    input_size = (256,256,1)
    inputs = Input(input_size)
    
    
    #Encoder
    
    conv1_1 = SeparableConv2D(64, (3,3), padding = 'same', activation = 'relu', 
                   kernel_initializer='he_normal')(inputs)   
    conv1_2 = SeparableConv2D(64, (3,3), padding = 'same', activation = 'relu', 
                   kernel_initializer='he_normal')(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1_2)
    
    
    conv2_1 = SeparableConv2D(128, (3,3), padding = 'same', activation = 'relu', 
                   kernel_initializer='he_normal')(pool1)
    conv2_2 = SeparableConv2D(128, (3,3), padding = 'same', activation = 'relu', 
                   kernel_initializer='he_normal')(conv2_1)
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2_2)
    
    
    conv3_1 = SeparableConv2D(256, (3,3), padding = 'same', activation = 'relu', 
                   kernel_initializer='he_normal')(pool2)
    conv3_2 = SeparableConv2D(256, (3,3), padding = 'same', activation = 'relu', 
                   kernel_initializer='he_normal')(conv3_1)
    pool3 = MaxPooling2D(pool_size=(2,2))(conv3_2)
    
    conv4_1 = SeparableConv2D(512, (3,3), padding = 'same', activation = 'relu', 
                   kernel_initializer='he_normal')(pool3)
    conv4_2 = SeparableConv2D(512, (3,3), padding = 'same', activation = 'relu', 
                   kernel_initializer='he_normal')(conv4_1)
    drop4_2 = Dropout(0.5)(conv4_2)
    pool4 = MaxPooling2D(pool_size=(2,2))(drop4_2)
    
    
    conv5_1 = SeparableConv2D(1024, (3,3), padding = 'same', activation = 'relu', 
                   kernel_initializer='he_normal')(pool4)
    conv5_2 = SeparableConv2D(1024, (3,3), padding = 'same', activation = 'relu', 
                   kernel_initializer='he_normal')(conv5_1)
    drop5_2 = Dropout(0.5)(conv5_2)
    
    
    #Decoder
    
    up6_1 = Conv2DTranspose(512, (2,2), strides = (2,2), padding = 'same', 
                          activation = 'relu', 
                          kernel_initializer='he_normal')(drop5_2)
    up6_2 = concatenate([drop4_2, up6_1], axis = 3)
    conv6_1 = SeparableConv2D(512, (3,3), padding = 'same', activation = 'relu', 
                   kernel_initializer='he_normal')(up6_2)
    conv6_2 = SeparableConv2D(512, (3,3), padding = 'same', activation = 'relu', 
                   kernel_initializer='he_normal')(conv6_1)
    
    
    up7_1 = Conv2DTranspose(256, (2,2), strides = (2,2), padding = 'same', 
                          activation = 'relu', 
                          kernel_initializer='he_normal')(conv6_2)
    up7_2 = concatenate([conv3_2, up7_1], axis = 3)
    conv7_1 = SeparableConv2D(256, (3,3), padding = 'same', activation = 'relu', 
                   kernel_initializer='he_normal')(up7_2)
    conv7_2 = SeparableConv2D(256, (3,3), padding = 'same', activation = 'relu', 
                   kernel_initializer='he_normal')(conv7_1)
    
    
    up8_1 = Conv2DTranspose(128, (2,2), strides = (2,2), padding = 'same', 
                          activation = 'relu', 
                          kernel_initializer='he_normal')(conv7_2)
    up8_2 = concatenate([conv2_2, up8_1], axis = 3)
    conv8_1 = SeparableConv2D(128, (3,3), padding = 'same', activation = 'relu', 
                   kernel_initializer='he_normal')(up8_2)
    conv8_2 = SeparableConv2D(128, (3,3), padding = 'same', activation = 'relu', 
                   kernel_initializer='he_normal')(conv8_1)
    
    
    up9_1 = Conv2DTranspose(64, (2,2), strides = (2,2), padding = 'same', 
                          activation = 'relu', 
                          kernel_initializer='he_normal')(conv8_2)
    up9_2 = concatenate([conv1_2, up9_1], axis = 3)
    conv9_1 = SeparableConv2D(64, (3,3), padding = 'same', activation = 'relu', 
                   kernel_initializer='he_normal')(up9_2)
    conv9_2 = SeparableConv2D(64, (3,3), padding = 'same', activation = 'relu', 
                   kernel_initializer='he_normal')(conv9_1)
    conv9_3 = SeparableConv2D(2, 3, activation = 'relu', padding = 'same', 
                   kernel_initializer = 'he_normal')(conv9_2)
    
  
    conv10 = SeparableConv2D(1, 1, activation = 'sigmoid')(conv9_3)
    
    if(vis):
        outputs = [conv1_1, conv1_2, pool1, conv2_1, conv2_2, pool2, conv3_1, 
                   conv3_2, pool3, conv4_1, conv4_2, drop4_2, pool4, conv5_1, 
                   conv5_2, drop5_2, up6_1, up6_2,conv6_1, conv6_2, up7_1, 
                   up7_2, conv7_1, conv7_2, up8_1, up8_2, conv8_1, conv8_2,
                   up9_1, up9_2, conv9_1, conv9_2, conv9_3, conv10]
        
        model = Model(inputs = [inputs], outputs = outputs)
        model.summary()
    else:
        model = Model(inputs = [inputs], outputs = [conv10])
        model.summary()
        
    model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, 
                  metrics=[dice_coef, 'accuracy'])
    
   
    return model

