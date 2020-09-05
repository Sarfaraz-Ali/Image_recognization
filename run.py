from data_load import data_load
from flatten import flatten 
from model import model
import numpy as np 
from skimage.transform import resize
from imageio import imread 
from predict import predict

def run() :
	
	
	X_train , Y_train , X_test , Y_test , classes = data_load()
	
	num_px = X_train.shape[1]

	X_train , X_test = flatten(X_train , X_test)

	d = model(X_train , Y_train , X_test , Y_test)

	image = "my_image.jpg"   # change this to the name of your image file 

	
	image = imread(image)
	image = image/255
	image = resize(image, output_shape=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
	my_predicted_image = predict(d["w"], d["b"], image)
	
	if "cat" == (classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") ) :
		print("It is a cat .")

	else :
		print("It is not a cat .")
