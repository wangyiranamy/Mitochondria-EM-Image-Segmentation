import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
from keras import backend as keras
from models.modelbase import ModelBase
from evaluate import dice_coef



def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

class Unet(ModelBase):
	def __init__(self,input_size,path):
		super(Unet, self).__init__(path)
		self.input_size = input_size
		self.build_model()
		self.model_checkpoint = ModelCheckpoint('saved_model/unet.hdf5',monitor='val_loss',verbose=1,save_best_only=True)

	def build_model(self):
		inputs = Input(self.input_size+(1,))
		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
		conv1 = BatchNormalization()(conv1)
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
		conv2 = BatchNormalization()(conv2)
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
		conv3 = BatchNormalization()(conv3)
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
		conv4 = BatchNormalization()(conv4)
		drop4 = Dropout(0.5)(conv4)
		pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

		conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
		conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
		conv5 = BatchNormalization()(conv5)
		drop5 = Dropout(0.5)(conv5)

		up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
		merge6 = concatenate([drop4,up6], axis = 3)
		conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
		conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
		conv6 = BatchNormalization()(conv6)

		up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
		merge7 = concatenate([conv3,up7], axis = 3)
		conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
		conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
		conv7 = BatchNormalization()(conv7)

		up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
		merge8 = concatenate([conv2,up8], axis = 3)
		conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
		conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
		conv8 = BatchNormalization()(conv8)

		up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
		merge9 = concatenate([conv1,up9], axis = 3)
		conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
		conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		conv9 = BatchNormalization()(conv9)
		conv10 = Conv2D(1, 1, activation = 'sigmoid',name='preds')(conv9)

		model = Model(input = inputs, output = conv10)

		model.compile(optimizer = Adam(lr = 1e-4), loss = dice_coef_loss, metrics = [dice_coef])
		self.model = model



class Attention_unet(ModelBase):
	def __init__(self,input_size,path):
		super(Attention_unet, self).__init__(path)

		self.input_size = input_size
		self.num_seg_class=1
		self.model_checkpoint = ModelCheckpoint('saved_model/attention_unet.hdf5',monitor='val_loss',verbose=1,save_best_only=True)
		self.build_model()

	def expend_as(self,tensor, rep):
		my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep})(tensor)
		return my_repeat

	def AttnGatingBlock(self,x, g, inter_shape):
		shape_x = K.int_shape(x)  # 32
		shape_g = K.int_shape(g)  # 16

		theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)  # 16
		shape_theta_x = K.int_shape(theta_x)

		phi_g = Conv2D(inter_shape, (1, 1), padding='same')(g)
		upsample_g = Conv2DTranspose(inter_shape, (3, 3),strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),padding='same')(phi_g)  # 16

		concat_xg = add([upsample_g, theta_x])
		act_xg = Activation('relu')(concat_xg)
		psi = Conv2D(1, (1, 1), padding='same')(act_xg)
		sigmoid_xg = Activation('sigmoid')(psi)
		shape_sigmoid = K.int_shape(sigmoid_xg)
		upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32

		upsample_psi = self.expend_as(upsample_psi, shape_x[3])
		y = multiply([upsample_psi, x])
		result = Conv2D(shape_x[3], (1, 1), padding='same')(y)
		result_bn = BatchNormalization()(result)
		return result_bn

	def UnetGatingSignal(self,input, is_batchnorm=False):
		shape = K.int_shape(input)
		x = Conv2D(shape[3] * 2, (1, 1), strides=(1, 1), padding="same")(input)
		if is_batchnorm:
			x = BatchNormalization()(x)
		x = Activation('relu')(x)
		return x

	def UnetConv2D(self,input, outdim, is_batchnorm=False):
		x = Conv2D(outdim, (3, 3), strides=(1, 1), padding="same")(input)
		if is_batchnorm:
			x =BatchNormalization()(x)
		x = Activation('relu')(x)

		x = Conv2D(outdim, (3, 3), strides=(1, 1), padding="same")(x)
		if is_batchnorm:
			x = BatchNormalization()(x)
		x = Activation('relu')(x)
		return x


	def build_model(self):
		inputs = Input(self.input_size+(1,))
		conv = Conv2D(16, (3, 3), padding='same')(inputs)  # 'valid'
		conv = LeakyReLU(alpha=0.3)(conv)

		conv1 = self.UnetConv2D(conv, 32,is_batchnorm=True)  # 32 128
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

		conv2 = self.UnetConv2D(pool1, 32,is_batchnorm=True)  # 32 64
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

		conv3 = self.UnetConv2D(pool2, 64,is_batchnorm=True)  # 64 32
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

		conv4 = self.UnetConv2D(pool3, 64,is_batchnorm=True)  # 64 16
		pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

		center = self.UnetConv2D(pool4, 128,is_batchnorm=True)  # 128 8

		gating = self.UnetGatingSignal(center, is_batchnorm=True)
		attn_1 = self.AttnGatingBlock(conv4, gating, 128)
		up1 = concatenate([Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same',activation="relu")(center), attn_1], axis=3)

		gating = self.UnetGatingSignal(up1, is_batchnorm=True)
		attn_2 = self.AttnGatingBlock(conv3, gating, 64)
		up2 = concatenate([Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same',activation="relu")(up1), attn_2], axis=3)

		gating = self.UnetGatingSignal(up2, is_batchnorm=True)
		attn_3 = self.AttnGatingBlock(conv2, gating, 32)
		up3 = concatenate([Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',activation="relu")(up2), attn_3], axis=3)

		up4 = concatenate([Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',activation="relu")(up3), conv1], axis=3)


		conv8 = Conv2D(self.num_seg_class + 1, (1, 1), activation='relu', padding='same')(up4)
		act =  Conv2D(1, (1,1), activation = 'sigmoid',name='preds')(conv8)

		model = Model(inputs=inputs, outputs=act)
		model.compile(optimizer='adam', loss = dice_coef_loss, metrics = [dice_coef])
		self.model = model
