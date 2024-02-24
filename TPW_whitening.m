function [test_feature_pca,query_feature_pca] = TPW_whitening(train_features_normalize,test_features_normalize,query_features,dim,TPW)
if nargin==5 
    %%%%%%%%%%%%%% TPW  %%%%%%%%%%%%%%%%%%%%%%%
    %%% PCAw %%%%
    [oxford_feature,coeff,mu,u,s]= pca_and_whitening(train_features_normalize,test_features_normalize,size(test_features_normalize,2));
    %%% Self_PCA %%%
    [coeff11,scoreTrain11,~,~,~,mu11]=pca(oxford_feature);
    test_feature_pca=scoreTrain11(:,1:dim);
    test_feature_pca=normalize(test_feature_pca,2,"norm");
    %%% Query_TPW %%%%
    q_features=normalize(query_features,2,"norm");
  
      if size(q_features,2)>size(coeff,2)
          p=size(coeff,2);
      else
          p=size(q_features,2);
      end
    query_features_white=query_pca(q_features,coeff,mu,u,s,p);

    q_features=(query_features_white-mu11)*coeff11;
    query_feature_pca=q_features(:,1:dim);
    
    query_feature_pca=normalize(query_feature_pca,2,"norm");
   
else
    %%%%%%%%%%%%%% PW %%%%%%%%%%%%%%%%%%%%%%%%%%%
    [oxford_feature,coeff,mu,u,s]= pca_and_whitening(train_features_normalize,test_features_normalize,dim);
    test_feature_pca=normalize(oxford_feature,2,"norm");
    
    q_features=normalize(query_features,2,"norm");
    query_features_white=query_pca(q_features,coeff,mu,u,s,dim);
    query_feature_pca=normalize(query_features_white,2,"norm");
end
end



