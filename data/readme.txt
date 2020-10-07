This is the unprocessed data loaded from database. Load it in MATLAB using

load('data.mat')


It contains 2 variables, input and output. Input is the features used to train. Each column denotes one feature. There are 10 columns which means 10 features. The row is the number of observation. The features are

1.      Credit score (In the dataset but not used for training)
2.      Combined LTV
3.      Debt to Income Ratio
4.      LTV
5.      Interest rate
6.      Current default status (1 or 0) (target output)
7.      Default count
8.      Mean CLDS
9.      Standard deviation CLDS
10.  	UPB value

The output variable is a column vector as the label for the features in input. 1 means default, 0 means non default. 

Run "Data_generation" to generate training and validation dataset.( From original data and SMOTE data).
