clear; clc; close all

load('..\data\data.mat');

% split up the data into default and no default groups
num_default = length(find(input_scale(:,6)>0));
num_no_default = length(find(input_scale(:,6)==0));

data_default = zeros(num_default,10);
data_no_default = zeros(num_no_default,10);
%%
c1 = 1; c2 = 1;
for i = 1:length(input_scale)
    if input_scale(i,6) > 0
        data_default(c1,:) = input_scale(i,:);
        c1 = c1 + 1;
    else
        data_no_default(c2,:) = input_scale(i,:);
        c2 = c2 + 1;
    end
end

% validation data
val = 1000;
val_data_default = data_default(1:val,:);
val_data_no_default = data_no_default(1:val,:);
val_data = [val_data_default ; val_data_no_default];

data_default = data_default(val+1:num_default,:);
data_no_default = data_no_default(val+1:num_no_default,:);

% trim this data to have 50/50
%  Check SMOTE
data_no_default = data_no_default(1:num_default,:);

% select features
features = [1,2,3,4,5,7,8,9,10];
features = [4,7,9];
trim_data_default = data_default(:,features); % 6
trim_data_no_default = data_no_default(:,features);

% fuzzy clustering
cnd = 2;                       % number of clusters
cd = 9;                       % number of clusters
Z_no_default = trim_data_no_default;    % N by n data matrix
Z_default = trim_data_default;    % N by n data matrix

options = [NaN 100 0.001 0];
[CENTER_no_default, U_no_default] = fcm(Z_no_default,cnd,options);
[CENTER_default, U_default] = fcm(Z_default,cd,options);
%%
clc

NNcount = 0;
badcount = 0;
goodcount = 0;

for j = 1:val*2
    test_dude = val_data(j,:);
    trim_test_dude = test_dude(features);
    
    for i = 1:cd % all clusters
        dist_default(i) = norm(CENTER_default(i,:) - trim_test_dude);       % closest default cluster
    end
    for i = 1:cnd % all clusters
        dist_no_default(i) = norm(CENTER_no_default(i,:) - trim_test_dude); % closest no default cluster
    end
    
    default    = 1/min(dist_default);
    no_default = 1/min(dist_no_default);
    
    
    predicted_default_flag = default > no_default;
    
    if abs(default - no_default) < 6
        NNcount = NNcount + 1;
        continue
    end
    
    if predicted_default_flag ~= test_dude(6)
        badcount = badcount + 1;
    else
        goodcount = goodcount + 1;
    end
    
end
disp('done')

NNcount
badcount 
goodcount

%%


