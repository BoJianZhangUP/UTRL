% Please download the dataset(oxford5k,paris6k,roxford,rparis) and repvgg model (imagenet_repvggplus_L2pse_deploy) first and put them in the  data’ folder.
% Image data and features can be downloaded via the following link：https://pan.baidu.com/s/1reESRGJkWRligvCFCjkkdg?pwd=xq6w Retrieve code：xq6w
extract_repvgg_features;

clear;
% parameters seting
net='repvgg';      
test_set='paris6k'; %% oxford5k, roxford, ......
layers='mul4_24';
datapath='../data/';
qe=10;
%%%%%%%% Import feature paths %%%%%%%%%%%
switch test_set
    case {'oxford5k'}
        train_set='paris6k';
    case {'paris6k'}
        train_set='oxford5k';
    case {'roxford'}
        train_set='rparis';
    case {'rparis'}
        train_set='roxford';
    case {'oxford105k'}
        train_set='paris6k';
        query_set='oxford5k';
        nquery_files = dir(fullfile(datapath,['oxford5k','_nquery_mul4_24'],'*.mat'));
    case {'paris106k'}
        train_set='oxford5k';
        query_set='paris6k';
        nquery_files = dir(fullfile(datapath,['paris6k','_nquery_mul4_24'],'*.mat'));
end

eval(['load gnd_' test_set '.mat']);
if ~exist("query_files","var")
       nquery_files = dir(fullfile(datapath,[test_set,'_nquery_mul4_24'],'*.mat'));
end
if  strcmpi(test_set,'oxford105k') || strcmpi(test_set,'paris106k')
    test_files = dir(fullfile(datapath,[test_set,'_mul4_24'],'*.mat'));
    nquery_files = dir(fullfile(datapath,[query_set,'_nquery_mul4_24'],'*.mat'));
else
    test_files = dir(fullfile(datapath,[test_set,'_mul4_24'],'*.mat'));
end
train_files = dir(fullfile(datapath,[train_set,'_mul4_24'],'*.mat'));

%%% Calculate the standard deviation of the feature maps %%%%
SD=standard_deviation(train_files);

%%%%% Use the UTRL method for test sets %%%%%%
eval(['load gnd_' test_set '.mat']);
test_features=extract_UTRL_features(test_files,imlist,SD);
test_features_normalize=normalize(test_features,2,"norm");

%%%%% Use the UTRL method for train sets %%%%%%
eval(['load gnd_' train_set '.mat']);
train_features=extract_UTRL_features(train_files,imlist,SD);
train_features_normalize=normalize(train_features,2,"norm");

%%%%% Use the UTRL method for query images %%%%%%
eval(['load gnd_' test_set '.mat']);
if ~exist("q_name","var")
    q_name=qimlist;
    qidx=priorindex_queries;
end
if size(nquery_files,1)==70
    query_nocrop_features=extract_UTRL_features(nquery_files,q_name,SD);
    query_nocrop_features_normalize=normalize(query_nocrop_features,2,"norm");
else
    query_nocrop_features_normalize=test_features_normalize(qidx,:);
end

warning off;

%%%%%%%%%%%% TPW and compute mAP %%%%%%%%%
if  strcmpi(test_set,'roxford5k') || strcmpi(test_set,'rparis6k')
    display_rquery_results(test_set,train_features_normalize,test_features_normalize,query_nocrop_features_normalize,gnd,qe);
else
    display_query_results(net,layers,test_set,train_features_normalize,test_features_normalize,query_nocrop_features_normalize,gnd,qe);
end

