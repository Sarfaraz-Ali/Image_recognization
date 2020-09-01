from data_load import data_load
from dim import dim
from flatten import flatten 
#from sigmoid import sigmoid
#from initialize import initialize 
#from propagate import propagate 
#from optimize import optimize
from predict import predict 
from model import model
import numpy as np 
from skimage.transform import resize
from PIL import Image
from scipy import ndimage
import imageio

def run() :
	
	
	X_train , Y_train , X_test , Y_test , classes = data_load()
	
	m_train , m_test , num_px = dim(X_train , X_test)

	X_train , X_test = flatten(X_train , X_test)

	d = model(X_train , Y_train , X_test , Y_test)




	my_image = "my_image.jpg"   # change this to the name of your image file 

	#  preprocess the image 
	
	fname = my_image
	image = np.array(imageio.imread(fname))
	image = image/255
	my_image = resize(image, output_shape=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
	#my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
	my_predicted_image = predict(d["w"], d["b"], my_image)

	if "cat" == (classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") ) :
		print("It is a cat .")

	else :
		print("It is not a cat .")