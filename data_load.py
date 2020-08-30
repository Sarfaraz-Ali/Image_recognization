
import h5py
import numpy as np 

def data_load() :
	train_data = h5py.File("train_catvnoncat.h5","r")
	test_data = h5py.File("test_catvnoncat.h5","r")

	train_set_x_orig = np.array(train_data["train_set_x"][:])
	train_set_y_orig = np.array(train_data["train_set_y"][:])

	test_set_x_orig = np.array(test_data["test_set_x"][:])
	test_set_y_orig = np.array(test_data["test_set_y"][:])

	classes = np.array(test_data["list_classes"][:])

	return (train_set_x_orig , train_set_y_orig , test_set_x_orig , test_set_y_orig , classes)

