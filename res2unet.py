from keras.models import *
from keras.layers import Input, Conv2D, UpSampling2D, BatchNormalization, Activation, add,average,concatenate
from keras.layers.core import Lambda
from keras.optimizers import *
from keras.losses import binary_crossentropy
#import tensorflow as tf
#import keras.backend as K
from metrics2 import iou_coeff
from metrics2 import recall
from metrics2 import dice_loss
from metrics2 import precision
from metrics2 import ACL5


IMG_SIZE = 256
h_heuns_method=0.5

def res_block(x, nb_filters, strides):
    res_path = BatchNormalization()(x)
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv2D(filters=nb_filters[0], kernel_size=(3, 3), padding='same' ,strides=strides[0])(res_path)

    res_path = BatchNormalization()(res_path)
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv2D(filters=nb_filters[1], kernel_size=(3, 3), padding='same',strides=strides[1])(res_path)
    hpath = Lambda(lambda x: x * h_heuns_method)(res_path)

    shortcut = Conv2D(nb_filters[1], kernel_size=(1, 1),strides=strides[0])(x)
    shortcut = BatchNormalization()(shortcut)

    res_path = add([shortcut, hpath])#suma corta
    return res_path
def res_block2(x,y,nb_filters, strides):
    res_path = BatchNormalization()(x)
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv2D(filters=nb_filters[0], kernel_size=(3, 3), padding='same' ,strides=strides[0])(res_path)

    res_path = BatchNormalization()(res_path)
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv2D(filters=nb_filters[1], kernel_size=(3, 3), padding='same',strides=strides[1])(res_path)
    hpath = Lambda(lambda x: x * h_heuns_method)(res_path)

    shortcut = Conv2D(nb_filters[1], kernel_size=(1, 1),strides=strides[0])(x)
    shortcut = BatchNormalization()(shortcut)

    res_path = add([shortcut, hpath])#suma corta

    res_path = average([y, res_path])#suma doble 
    return res_path


def encoder(x):
    to_decoder = []

    main_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1))(x)
    main_path = BatchNormalization()(main_path)
    main_path = Activation(activation='relu')(main_path)

    main_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1))(main_path)
    hpath = Lambda(lambda x: x * h_heuns_method)(main_path)

    shortcut = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1))(x)
    shortcut = BatchNormalization()(shortcut)

    main_path = add([shortcut, hpath])#suma corta

    to_decoder.append(main_path)


    s1 = Conv2D(filters=128, kernel_size=(1, 1), strides=(2, 2))(x)
    s1 = BatchNormalization()(s1)

    main_path = res_block2(main_path,s1, [128, 128], [(2, 2), (1, 1)]) 
    to_decoder.append(main_path)

    main_path = res_block(main_path, [256, 256], [(2, 2), (1, 1)])
    to_decoder.append(main_path)

    s2 = Conv2D(filters=512, kernel_size=(1, 1), strides=(4, 4))(to_decoder[1])
    s2 = BatchNormalization()(s2)

    main_path = res_block2(main_path,s2, [512, 512], [(2, 2), (1, 1)])
    to_decoder.append(main_path)

    return to_decoder


def decoder(x, from_encoder):
    main_path = UpSampling2D(size=(2, 2))(x)#32x32
    main_path1 = concatenate([main_path, from_encoder[3]], axis=3)
    main_path = res_block(main_path1, [512, 512], [(1, 1), (1, 1)])

    main_path = UpSampling2D(size=(2, 2))(main_path)###64x64
    main_path = concatenate([main_path, from_encoder[2]], axis=3)#
    u1 = UpSampling2D(size=(2, 2))(main_path1)#
    u1 = Conv2D(256, kernel_size=(1, 1),strides=(1, 1))(u1)
    u1 = BatchNormalization()(u1)
    main_path = res_block2(main_path,u1, [256, 256], [(1, 1), (1, 1)])#

    main_path = UpSampling2D(size=(2, 2))(main_path)#128x128
    main_path2 = concatenate([main_path, from_encoder[1]], axis=3)#
    main_path = res_block(main_path2, [128, 128], [(1, 1), (1, 1)])#

    main_path = UpSampling2D(size=(2, 2))(main_path)#256x256
    main_path = concatenate([main_path, from_encoder[0]], axis=3)#256x256

    u2 = UpSampling2D(size=(2,2))(main_path2)#
    u2 = Conv2D(64, kernel_size=(1, 1),strides=(1, 1))(u2)#
    u2 = BatchNormalization()(u2)
    main_path = res_block2(main_path,u2, [64, 64], [(1, 1), (1, 1)])#256x256

    return main_path


def res2unet(lrate=8.00E-05,pretrained_weights=None):
    print(lrate)
    input_size=(IMG_SIZE, IMG_SIZE, 3)
    inputs = Input(shape=input_size)

    to_decoder = encoder(inputs)

    path = res_block(to_decoder[3], [1024, 1024], [(2, 2), (1, 1)])####bridge

    path = decoder(path, from_encoder=to_decoder)


    path = Conv2D(2, kernel_size=(3, 3),activation='relu', padding='same', strides=(1, 1))(path)
    path = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(path)
    model = Model(input=inputs, output=path)
    model.compile(optimizer=Adam(lr=lrate), loss=ACL5, metrics=[dice_loss,iou_coeff,precision,recall])
    model.summary()
    if (pretrained_weights):
        model.load_weights(pretrained_weights)
    return model