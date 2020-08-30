	 
def flatten(train_set_x_orig , test_set_x_orig) :
	train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
	test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

	train_set_x = train_set_x_flatten/255
	test_set_x = test_set_x_flatten/255

	return (train_set_x , test_set_x)