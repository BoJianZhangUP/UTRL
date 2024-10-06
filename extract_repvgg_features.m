% Defining the path
clear
oxf5k_folderPath = fullfile('data', 'oxford5k_mul4_24');
% Check if the folder exists
if exist(oxf5k_folderPath, 'dir') && numel(dir(oxf5k_folderPath)) > 2
    fprintf('Folder %s already exists。\n', oxf5k_folderPath);
else
    % If it does not exist, create the folder
    mkdir(oxf5k_folderPath);
    fprintf('Folder %s Created。\n', oxf5k_folderPath);

    fprintf('Extracting oxford5k features ...\n')
    load('data\imagenet_repvggplus_L2pse_deploy.mat') % Loading Models
    layer='mul4_24';
    img_datasetPath = dir("data\oxford5k_image\*.jpg");
    minsize=224;
    parfor i=1:length(img_datasetPath)
        imgPath = [img_datasetPath(i).folder,'\',img_datasetPath(i).name];
        im = imread(imgPath);
        [h,w,~]=size(im);
        if w<minsize || h<minsize
            im = imresize(im, minsize/min(h,w));
        end

        features = activations(net,im,layer,'OutputAs','channels');
        parsave(['data\oxford5k_mul4_24\',erase(img_datasetPath(i).name,'.jpg'),'.mat'],features);

        % Dynamic Printing Progress
        if mod(i, 1000) == 0
            fprintf('Processed: %d\n', i);
        end
    end
    fprintf('Processing complete!\n');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
par6k_folderPath = fullfile('data', 'paris6k_mul4_24');

if exist(par6k_folderPath, 'dir') && numel(dir(par6k_folderPath)) > 2
    fprintf('Folder %s already exists。\n', par6k_folderPath);

else

    mkdir(par6k_folderPath);
    fprintf('Folder %s Created。\n', par6k_folderPath);

    fprintf('Extracting paris6k features ...\n')

    load('data\imagenet_repvggplus_L2pse_deploy.mat') % Loading Models
    layer='mul4_14';
    img_datasetPath = dir("data\paris6k_image\*.jpg");
    minsize=224;
    parfor i=1:length(img_datasetPath)
        imgPath = [img_datasetPath(i).folder,'\',img_datasetPath(i).name];
        im = imread(imgPath);
        [h,w,~]=size(im);
        if w<minsize || h<minsize
            im = imresize(im, minsize/min(h,w));
        end

        features = activations(net,im,layer,'OutputAs','channels');
        parsave(['data\paris6k_mul4_24\',erase(img_datasetPath(i).name,'.jpg'),'.mat'],features);

        % Dynamic Printing Progress
        if mod(i, 1000) == 0
            fprintf('Processed: %d\n', i);
        end
    end
    fprintf('Processing complete!\n');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
roxford_folderPath = fullfile('data', 'roxford_mul4_24');

if exist(roxford_folderPath, 'dir') && numel(dir(roxford_folderPath)) > 2
    fprintf('Folder %s already exists。\n', roxford_folderPath);

else

    mkdir(roxford_folderPath);
    fprintf('Folder %s Created。\n', roxford_folderPath);

    fprintf('Extracting roxford features ...\n')

    load('data\imagenet_repvggplus_L2pse_deploy.mat') % Loading Models
    layer='mul4_14';
    img_datasetPath = dir("data\roxford5k_image\*.jpg");
    minsize=224;
    parfor i=1:length(img_datasetPath)
        imgPath = [img_datasetPath(i).folder,'\',img_datasetPath(i).name];
        im = imread(imgPath);
        [h,w,~]=size(im);
        if w<minsize || h<minsize
            im = imresize(im, minsize/min(h,w));
        end

        features = activations(net,im,layer,'OutputAs','channels');
        parsave(['data\roxford_mul4_24\',erase(img_datasetPath(i).name,'.jpg'),'.mat'],features);

        % Dynamic Printing Progress
        if mod(i, 1000) == 0
            fprintf('Processed: %d\n', i);
        end
    end
    fprintf('Processing complete!\n');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
rparis_folderPath = fullfile('data', 'rparis_mul4_24');

if exist(rparis_folderPath, 'dir') && numel(dir(rparis_folderPath)) > 2
fprintf('Folder %s already exists。\n', rparis_folderPath);
    
else
    
    mkdir(rparis_folderPath);
    fprintf('Folder %s Created。\n', rparis_folderPath);

    fprintf('Extracting rparis features ...\n')

    load('data\imagenet_repvggplus_L2pse_deploy.mat') % Loading Models
    layer='mul4_14';
    img_datasetPath = dir("data\rparis6k_image\*.jpg");
    minsize=224;
    parfor i=1:length(img_datasetPath)
        imgPath = [img_datasetPath(i).folder,'\',img_datasetPath(i).name];
        im = imread(imgPath);
        [h,w,~]=size(im);
        if w<minsize || h<minsize
            im = imresize(im, minsize/min(h,w));
        end

        features = activations(net,im,layer,'OutputAs','channels');
        parsave(['data\rparis_mul4_24\',erase(img_datasetPath(i).name,'.jpg'),'.mat'],features);

        % Dynamic Printing Progress
        if mod(i, 1000) == 0
            fprintf('Processed: %d\n', i);
        end
    end
    fprintf('Processing complete!\n');
end