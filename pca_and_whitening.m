function [oxford_feature,coeff,mu,u,s,idx] = pca_and_whitening(XTrain1,XText,dim)

[coeff,scoreTrain,~,~,explained,mu]=pca(XTrain1);

% 

if nargin==2
    sum_explained = 0;
    idx = 0;
    while sum_explained < 99
        idx = idx + 1;
        sum_explained = sum_explained + explained(idx);
    end
    dim=idx;

end
if dim>size(scoreTrain,2)
    dim=min(dim,size(scoreTrain,2));
end
x_train=scoreTrain(:,1:dim);

%计算协方差矩阵
sigma=cov(x_train,'omitrows');
[u,s,~]=svd(sigma);
%用训练集的主成分分数来转换测试的数据集

% scoreTest=(XText-0.6*mu)*coeff;
scoreTest=(XText-mu)*coeff;
x_test=scoreTest(:,1:dim);


xRot=x_test*u;

% epsilon=1*10^(-4);
epsilon=1*10^(-5);
xPCAWhite=diag(1./(sqrt(diag(s)+epsilon)))*xRot';
oxford_feature=xPCAWhite';

end


