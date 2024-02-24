function features=extrsct_features(files,list,SD)
features=[];

parfor i=1:size(files,1)
    path=[files(i).folder,'\',list{i},'.mat'];
    fea_layers = importdata(path);
    feature = apply_UTRL_aggregation(fea_layers,SD);
    features = [features;feature];
    if mod(i,1000) == 0
        i
    end
end

end

