% Generating dataset for training and validation
clear; clc; close all

load('data.mat');

% split up the data into default and no default groups
num_default = length(find(input_scale(:,6)>0));
num_no_default = length(find(input_scale(:,6)==0));

data_default=input_scale(find(input_scale(:,6)>0),:);
data_no_default=input_scale(find(input_scale(:,6)==0),:);

% choose proportion of data for validation
percentage = 0.3;

% validation data
val_default = ceil(percentage*num_default);
val_no_default = ceil(percentage*num_no_default);
val_data_default = data_default(1:val_default,:);
val_data_no_default = data_no_default(1:val_no_default,:);

data_default = data_default(val_default+1:num_default,:);
data_no_default = data_no_default(val_no_default+1:num_no_default,:);

training_dataset = [data_default;data_no_default];
training_dataset = training_dataset(randperm(length(training_dataset)),:);
save('training_dataset.mat','training_dataset');

validation_dataset = [val_data_default;val_data_no_default];
validation_dataset = validation_dataset(randperm(length(validation_dataset)),:);
save('validation_dataset.mat','validation_dataset');

%% SMOTE training_dataset
[training_dataset_feature,training_dataset_label] = SMOTE ([training_dataset(:,1:5) training_dataset(:,7:10)],training_dataset(:,6));
SMOTE_training_dataset = [training_dataset_feature(:,1:5) training_dataset_label training_dataset_feature(:,6:9)];
save('SMOTE_training_dataset.mat','SMOTE_training_dataset');
%% SMOTE validation dataset
[validation_dataset_feature,validation_dataset_label] = SMOTE([validation_dataset(:,1:5) validation_dataset(:,7:10)],validation_dataset(:,6));
SMOTE_validation_dataset = [validation_dataset_feature(:,1:5) validation_dataset_label validation_dataset_feature(:,6:9)];
save('SMOTE_validation_dataset.mat','SMOTE_validation_dataset');