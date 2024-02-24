function []=display_rquery_results(test_set,train_features_normalize,test_features_normalize,query_nocrop_features_normalize,gnd,qe)
for m=1:8
    dim=8*2^(m-1);
    if m==8 && size(test_features_normalize,2)~=512
        dim=640;
    end
   
    %%%%%%%% PW %%%%%%%
    [PW_test_features_pca,PW_query_nocrop_features_pca]=TPW_whitening(train_features_normalize,test_features_normalize,query_nocrop_features_normalize,dim);
    %%%%%%%% TPW %%%%%%
    [TPW_test_features_pca,TPW_query_nocrop_features_pca]=TPW_whitening(test_set,train_features_normalize,test_features_normalize,query_nocrop_features_normalize,dim,'TPW');
 
    PW_dist=pdist2(PW_test_features_pca,PW_query_nocrop_features_pca,'euclidean');
    [~, PW_ranks] = sort(PW_dist, 'ascend');
    compute_r_map(dim,gnd,PW_ranks,test_set,'nocrop','PW_map');

    [PWranks_QE] = rank_qe(PW_test_features_pca', PW_query_nocrop_features_pca', PW_ranks,qe);
    compute_r_map(dim,gnd,PWranks_QE,test_set,'nocrop','PW_qe_map');

    dist=pdist2(TPW_test_features_pca,TPW_query_nocrop_features_pca,'euclidean');
    [~, TPWcrop_ranks] = sort(dist, 'ascend');
    compute_r_map(dim,gnd,TPWcrop_ranks,test_set,'nocrop','TPW_map');
   
    [TPWranks_QE] = rank_qe(TPW_test_features_pca', TPW_query_nocrop_features_pca', TPWcrop_ranks,qe);
    compute_r_map(dim,gnd,TPWranks_QE,test_set,'nocrop','TPW_qe_map');
 
     
end

end

