PCA_Clustering_ANFIS.m
	Test the two layer classification system using the data reduced by PCA.
	Tuneable parameters:
		cd	Number of default data centres
		cnd Number of non default data centres
		threshold Threshold to decide ambiguous data that are sent tO ANFIS
	Overall ROC curve, precision and sensitivity are checked.


SPCA_Clustering_ANFIS.m
	Test the two layer classification system using the data reduced by sparse PCA.
	Change the cardinality( Number of sparse variable in the projection matrix) in line 26.
	Tuneable parameters:
		cd	Number of default data centres
		cnd Number of non default data centres
		threshold Threshold to decide ambiguous data that are sent to ANFIS
	Overall ROC curve, precision and sensitivity are checked.