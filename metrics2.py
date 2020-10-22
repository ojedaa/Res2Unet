from keras import backend as K
from keras.losses import binary_crossentropy
import tensorflow as tf
import numpy as np
def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score
def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss
def iou_coeff(y_true, y_pred):
    smooth=1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union=K.sum(y_true_f) + K.sum(y_pred_f)-intersection
    mvalue=(intersection+smooth)/(union+smooth)
    return mvalue
def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
def ACL5(y_true, y_pred): 

	#y_pred = K.cast(y_pred, dtype = 'float64')

	print(K.int_shape(y_pred))

	x = y_pred[:,1:,:,:] - y_pred[:,:-1,:,:] # horizontal and vertical directions 
	y = y_pred[:,:,1:,:] - y_pred[:,:,:-1,:]

	delta_x = x[:,1:,:-2,:]**2
	delta_y = y[:,:-2,1:,:]**2
	delta_u = K.abs(delta_x + delta_y) 

	epsilon = 0.00000001 # where is a parameter to avoid square root is zero in practice.
	w = 1####
	lenth = w * K.sum(K.sqrt(delta_u + epsilon)) # equ.(11) in the paper


	C_1 = np.ones((256, 256))
	C_2 = np.zeros((256, 256))

	region_in = K.abs(K.sum( y_pred[:,:,:,0] * ((y_true[:,:,:,0] - C_1)**2) ) ) # equ.(12) in the paper
	region_out = K.abs(K.sum( (1-y_pred[:,:,:,0]) * ((y_true[:,:,:,0] - C_2)**2) )) # equ.(12) in the paper

	lambdaP = 5 # lambda parameter could be various.
	
	loss =  lenth + lambdaP * ((region_in) + (region_out)) 

	return loss