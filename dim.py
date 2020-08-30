
def dim(train_set_x_orig , test_set_x_orig) :
	
	m_train = train_set_x_orig.shape[0]
	m_test = test_set_x_orig.shape[0]
	num_px = train_set_x_orig.shape[1]

	return ( m_train , m_test , num_px )


