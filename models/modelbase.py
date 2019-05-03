
class ModelBase(object):

	def __init__(self,path):
		self.path = path
		self.model = None
		self.model_checkpoint = None

	def train(self,data_gen,val_gen,steps_per_epoch=500,epochs=50):
		return self.model.fit_generator(data_gen,steps_per_epoch=steps_per_epoch,epochs=epochs,validation_data=val_gen,validation_steps=20,callbacks=[self.model_checkpoint])
	def predict(self,data_gen):
		return self.model.predict(data_gen,verbose=1)
		# return self.model.predict_generator(data_gen,10,verbose = 1)
	def load(self, name):
		print('-'*20)
		print("Loading model checkpoint ...\n")
		self.model.load_weights(self.path+ name+'.hdf5')
		print("Model loaded")

	def build_model(self):

		raise NotImplementedError
