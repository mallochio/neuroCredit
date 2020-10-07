Cluster_test.m
	A test file for the clustering classifier

Cluster_test_anfis.m
	clustering classifier with ANFIS in cascade

Adjustable parameters:
	Clustering
		features
		cd
		cnd
		options
		threshold
	ANFIS
		features_anfis
		numMFs
		mfType
		epoch_n

Fmatrix.mat
	all possible feature combinations in an 255 x 8 matrix

lookup.mat and lookup1.mat
	rows are defined as: [cnd, cd, row_in_Fmatrix, NNcount, badcount, goodcount, reward]
	options = [NaN 100 0.001 0]   and   threshold = 4
	reward = (goodcount + NNcount*0.80)/(goodcount + NNcount + badcount);