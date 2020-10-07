% 1. Original proportion.
% 2. Separate dataset of training and validation used
% 3. Number of default and non default centers and threshold tunable
% 4. NO PCA

clear; clc; close all

% load('..\data\training_dataset');
load('..\data\SMOTE_training_dataset');

% training data
num_default = length(find(training_dataset(:,6)>0));
num_no_default = length(find(training_dataset(:,6)==0));

data_default=training_dataset(find(training_dataset(:,6)>0),:);
data_no_default=training_dataset(find(training_dataset(:,6)==0),:);

% select features
features = [4,7,9];
trim_data_default = data_default(:,features); % label at column 6
trim_data_no_default = data_no_default(:,features);

% fuzzy clustering
cd = 9;                       % number of default clusters
cnd = 2;                       % number of non default clusters

Z_no_default = trim_data_no_default;    % N by n data matrix
Z_default = trim_data_default;    % N by n data matrix

options = [NaN 100 0.001 0];
[CENTER_no_default, U_no_default] = fcm(Z_no_default,cnd,options);
[CENTER_default, U_default] = fcm(Z_default,cd,options);

clc
%%
%---- find outliers from training data
threshold = 4;

dist_default_dcenter_matrix = zeros(cd,num_default);
dist_default_ndcenter_matrix = zeros(cnd,num_default);
dist_no_default_dcenter_matrix = zeros(cd,num_no_default);
dist_no_default_ndcenter_matrix = zeros(cnd,num_no_default);
locat_unsure_default_data = [];
locat_unsure_no_default_data = [];

for j = 1:num_default
    
    for i = 1:cd % all clusters
        dist_default_dcenter_matrix(i,j) = norm(CENTER_default(i,:) - trim_data_default(j,:));
    end
    for i = 1:cnd % all clusters
        dist_default_ndcenter_matrix(i,j) = norm(CENTER_no_default(i,:) - trim_data_default(j,:));
    end
    
    default    = 1/min(dist_default_dcenter_matrix(:,j));
    no_default = 1/min(dist_default_ndcenter_matrix(:,j));
    
    if abs(default - no_default) < threshold
        locat_unsure_default_data = [locat_unsure_default_data;j];
    end
    
end

for j = 1:num_no_default
    
    for i = 1:cnd % all clusters
        dist_no_default_ndcenter_matrix(i,j) = norm(CENTER_no_default(i,:) - trim_data_no_default(j,:)); 
    end
    for i = 1:cd % all clusters
        dist_no_default_dcenter_matrix(i,j) = norm(CENTER_default(i,:) - trim_data_no_default(j,:)); 
    end
    
    default    = 1/min(dist_no_default_dcenter_matrix(:,j));
    no_default = 1/min(dist_no_default_ndcenter_matrix(:,j));
    
    if abs(default - no_default) < threshold
        locat_unsure_no_default_data = [locat_unsure_no_default_data;j];
    end
    
end

data_unsure = [data_default(locat_unsure_default_data,:);data_no_default(locat_unsure_no_default_data,:)];

%%
% select features for anfis training
% Original dataset: [2,4,7,9] 91.89% / [2,4,7,8,9,10] 92%/ [2,4,7,8,9] 90.60%
% SMOTE dataset: [2,4,7,9] 86.58% / [2,4,7,8,9,10] / [2,4,7,8,9] 86.64% less accuracy with better ROC curve!!
features_anfis = features;  %[2,4,7,8,9];
trim_data_unsure = data_unsure(:,features_anfis);
% anfis for unsure data
label_trim_data_unsure = [ones(length(locat_unsure_default_data),1);zeros(length(locat_unsure_no_default_data),1)];
trnData = [trim_data_unsure label_trim_data_unsure];
shuffle_trnData = trnData(randperm(length(trnData)),:);
numMFs = 2;
mfType = 'gbellmf';
epoch_n = 20;
in_fis = genfis1(shuffle_trnData,numMFs,mfType);
out_fis = anfis(shuffle_trnData,in_fis,epoch_n);

pred_trim_data_unsure=evalfis(shuffle_trnData(:,1:end-1),out_fis);
error_trim_data_unsure=pred_trim_data_unsure-shuffle_trnData(:,end);
vaf=max(0,1-(error_trim_data_unsure'*error_trim_data_unsure)/(shuffle_trnData(:,end)'*shuffle_trnData(:,end)))
% vaf of the anfis after training

%% validation

% load('..\data\validation_dataset');
load('..\data\SMOTE_validation_dataset');

% validation data
val_default = length(find(validation_dataset(:,6)>0));
val_no_default = length(find(validation_dataset(:,6)==0));

val_data_default = validation_dataset(find(validation_dataset(:,6)>0),:);
val_data_no_default = validation_dataset(find(validation_dataset(:,6)==0),:);
val_data = [val_data_default;val_data_no_default];

NNcount = 0;
badcount = 0;
goodcount = 0;
locat_unsure_val_data = [];

% selecting features and decreasing dimensions for validation data
trim_val_data_default = val_data_default(:,features);
trim_val_data_no_default = val_data_no_default(:,features);
trim_val_data = [trim_val_data_default val_data_default(:,6);trim_val_data_no_default val_data_no_default(:,6)];

for j = 1:val_default+val_no_default
    trim_test_dude = trim_val_data(j,1:end-1);
    
    for i = 1:cnd % all clusters
        dist_no_default(i) = norm(CENTER_no_default(i,:) - trim_test_dude); % closest no default cluster
    end
    for i = 1:cd % all clusters
        dist_default(i) = norm(CENTER_default(i,:) - trim_test_dude);       % closest default cluster
    end
    
    default    = 1/min(dist_default);
    no_default = 1/min(dist_no_default);
    
    predicted_default_flag = default > no_default;
    
    if abs(default - no_default) < threshold
        NNcount = NNcount + 1;
        locat_unsure_val_data=[locat_unsure_val_data;j];
        continue
    end
    
    if predicted_default_flag ~= trim_val_data(j,end)
        badcount = badcount + 1;
    else
        goodcount = goodcount + 1;
    end
    
end

%%
% Predict default status of outliers using anfis
test_NN_dude = val_data(locat_unsure_val_data,:);
trim_test_NN_dude = test_NN_dude(:,features_anfis);
label_test_NN_dude = val_data(locat_unsure_val_data,6);

pred_trim_test_NN_dude = evalfis(trim_test_NN_dude,out_fis);
NNgoodcount = 0;
NNbadcount = 0;
for j = 1:length(locat_unsure_val_data)
    if abs(pred_trim_test_NN_dude(j) - label_test_NN_dude(j)) < 0.5
        NNgoodcount = NNgoodcount + 1;
    else
        NNbadcount = NNbadcount + 1;
    end
end

accuracy = 1-(badcount+NNbadcount)/(val_default+val_no_default);

disp('done')

badcount 
goodcount
NNcount
NNgoodcount
NNbadcount
sprintf('The accuracy is %2.5f%%',accuracy*100)

%% test
comp_pred_label_NN_dude = [label_test_NN_dude pred_trim_test_NN_dude];
shuffle_comp_pred_label_NN = comp_pred_label_NN_dude(randperm(length(comp_pred_label_NN_dude)),:);
n = length(shuffle_comp_pred_label_NN);
figure(1);
plot(1:n,shuffle_comp_pred_label_NN(:,1),'go','MarkerSize',8,'LineWidth',3);grid on;
hold on; plot(1:n,shuffle_comp_pred_label_NN(:,2),'bx','MarkerSize',6,'LineWidth',0.5);
hold on; plot(linspace(1,n,500),0.5*ones(500),'--r','LineWidth',2);
legend('Real label of unsure data','Prediction of ANFIS');title('Validation of ANFIS');

% ROC curve matlab
figure(2);
[X,Y] = perfcurve(shuffle_comp_pred_label_NN(:,1),shuffle_comp_pred_label_NN(:,2),1);
plot(X,Y);
xlabel('False positive rate');
ylabel('True positive rate');
title('ROC for Classification by ANFIS');

% ROC curve homemade
threshold_classifier = linspace(0,1,1000);
X = zeros(1001,1);
Y = zeros(1001,1);
Z = zeros(1001,1);
TP = 0; TN = 0; FP = 0; FN = 0;
for i = 1:1000
    for j = 1:length(shuffle_comp_pred_label_NN)
        class_pred_trim_test_NN_dude = shuffle_comp_pred_label_NN(j,2) > threshold_classifier(i);
        if shuffle_comp_pred_label_NN(j,1) == 1 && class_pred_trim_test_NN_dude == 1
            TP = TP + 1;
        elseif shuffle_comp_pred_label_NN(j,1) == 1 && class_pred_trim_test_NN_dude == 0
            FN = FN + 1;
        elseif shuffle_comp_pred_label_NN(j,1) == 0 && class_pred_trim_test_NN_dude == 1
            FP = FP + 1;
        else
            TN = TN + 1;
        end
    end
    X(i) = FP/(TN + FP);
    Y(i) = TP/(TP + FN);
    Z(i) = TP/(TP + FP); % Precision
    TP = 0; TN = 0; FP = 0; FN = 0;
end
figure(3);
plot(X,Y,'x','MarkerSize',6,'LineWidth',2);
xlabel('False positive rate');
ylabel('True positive rate');
title('ROC for Classification by ANFIS');
sprintf('Taking the threshold as 0.5, the precision is %2.5f%%\n',Z(501)*100)
sprintf('The sensitivity is %2.5f%%\n',Y(501)*100)

min_dist_default_dcenter = min(dist_default_dcenter_matrix);
min_dist_default_ndcenter = min(dist_default_ndcenter_matrix);
min_dist_default = [min_dist_default_dcenter;min_dist_default_ndcenter];

min_dist_no_default_dcenter = min(dist_no_default_dcenter_matrix);
min_dist_no_default_ndcenter = min(dist_no_default_ndcenter_matrix);
min_dist_no_default = [min_dist_no_default_dcenter;min_dist_no_default_ndcenter];

diff_dist_default = min_dist_default(1,:)-min_dist_default(2,:);
num_default_closer_ndcenter = length(find(diff_dist_default>0));
diff_dist_no_default = min_dist_no_default(1,:)-min_dist_no_default(2,:);
num_no_default_closer_dcenter = length(find(diff_dist_no_default<0));

num_default_closer_ndcenter
num_no_default_closer_dcenter