function []=display_query_results(net,layers,test_set,train_features_normalize,test_features_normalize,query_nocrop_features_normalize,gnd,qe)
for m=1:8
    dim=8*2^(m-1);
    if m==8 && size(test_features_normalize,2)~=512
       dim=640;
    end

        %%%%%%%% PW %%%%%%%
        [PW_test_features_pca,PW_query_nocrop_features_pca]=TPW_whitening(train_features_normalize,test_features_normalize,query_nocrop_features_normalize,dim);
        %%%%%%%% TPW %%%%%%
        [TPW_test_features_pca,TPW_query_nocrop_features_pca]=TPW_whitening(train_features_normalize,test_features_normalize,query_nocrop_features_normalize,dim,'TPW');
       
        
        PW_dist=pdist2(PW_test_features_pca,PW_query_nocrop_features_pca,'euclidean');
        [~, PW_ranks] = sort(PW_dist, 'ascend');
        [PW_map,~] = compute_map (PW_ranks, gnd);
        
        [ranks_QE] = rank_qe(PW_test_features_pca', PW_query_nocrop_features_pca', PW_ranks,qe);
        [PW_qe_map,~] = compute_map (ranks_QE, gnd);
        
        dist=pdist2(TPW_test_features_pca,TPW_query_nocrop_features_pca,'euclidean');

        [~, TPWcrop_ranks] = sort(dist, 'ascend');
        [TPW_map,~] = compute_map (TPWcrop_ranks, gnd);
        
        [TPWranks_QE] = rank_qe(TPW_test_features_pca', TPW_query_nocrop_features_pca', TPWcrop_ranks,qe);
        [TPW_qe_map,~] = compute_map (TPWranks_QE, gnd);
     
  
    fprintf(['>> %s: %s: %s: %d dim:\n ' ...
        ' nocrop:PW_map:%.4f,PW_qe_map:%.4f,TPW_map:%.4f,TPW_qe_map:%.4f\n\n'], net,layers,test_set, dim, ...
        PW_map,PW_qe_map,TPW_map,TPW_qe_map);  
end

end

