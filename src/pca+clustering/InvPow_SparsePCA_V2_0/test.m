% Create synthetic data set
n = 1500; p = 500;
t = linspace(0, 1, p);
pc1 = max(0, (t - 0.5)> 0)';
pc2 = 0.8*exp(-(t - 0.5).^2/5e-3)';
pc3 = 0.4*exp(-(t - 0.15).^2/1e-3)' + 0.4*exp(-(t - 0.85).^2/1e-3)';
X = [pc1 + randn(p,1) pc2 + randn(p,1)...
  pc3 + randn(p,1)];

[coeff_X,~,latent_X] = pca(X);
X1=X*coeff_X;

F = sparsePCA(X, 3, 3, 0, 1);
X2=X*F

figure(1)
plot(t, [pc1 pc2 pc3]); axis([0 1 -1.2 1.2]);
title('Noiseless data');
figure(2);
plot(t, X);  axis([0 1 -6 6]);
title('Data + noise');
figure(3);
plot(t, X1);  %axis([0 1 -1.2 1.2]);
title('PCA');
figure(4)
plot(t, X2);  %axis([0 1 -1.2 1.2]);
title('SPCA');

%%
input_feature = [input_scale(:,2:5) input_scale(:,7:9)];
trim_input_feature = input_feature(1:10000,:);
[F,adj_var,cum_var] = sparsePCA(trim_input_feature, 2, 4, 0, 1);