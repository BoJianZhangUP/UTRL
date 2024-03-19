function [test_pca_features,coeff,mu,u,s,idx] = pca_and_whitening(XTrain,XText,dim)
% PCA and Whitening
% Inputs:
%   XTrain: Training data matrix
%   XText: Test data matrix
%   dim: Desired dimensionality after PCA
% Outputs:
%   test_pca_features: PCA-whitened features of the test data
%   coeff: Principal component coefficients
%   mu: Mean of the training data
%   u: Whitening matrix
%   s: Singular values of the covariance matrix
%   idx: Selected dimensionality after PCA based on explained variance

[coeff,scoreTrain,~,~,explained,mu]=pca(XTrain);
if nargin==2
    sum_explained = 0;
    idx = 0;
    while sum_explained < 99
        idx = idx + 1;
        sum_explained = sum_explained + explained(idx);
    end
    dim=idx;
end

% Limit dimensionality to the available dimensions
if dim>size(scoreTrain,2)
    dim=min(dim,size(scoreTrain,2));
end
x_train=scoreTrain(:,1:dim);

% Calculate covariance matrix and perform singular value decomposition
sigma=cov(x_train,'omitrows');
[u,s,~]=svd(sigma);

% Project test data onto the PCA space of training data
scoreTest=(XText-mu)*coeff;
x_test=scoreTest(:,1:dim);

% Whiten the PCA-projected test data
xRot=x_test*u;
epsilon=1*10^(-5);
xPCAWhite=diag(1./(sqrt(diag(s)+epsilon)))*xRot';
test_pca_features=xPCAWhite';

end
