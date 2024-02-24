function SD = standard_deviation(files)
sum_pool5=[];

parfor i=1:size(files,1)
    files_path=[files(i).folder,'\',files(i).name];
    pool5 = importdata(files_path);

    [~,~,channel] = size(pool5);

    d = reshape(sum(pool5,[1,2]),[1,channel]);

    sum_pool5=[sum_pool5;d];
end
    SD=std(sum_pool5,0,1);

end
