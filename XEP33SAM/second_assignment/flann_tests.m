addpath ('/home/marik0/Desktop/phd_courses/XEP33SAM/flann_matlab')

%fid = fopen('/home/marik0/Desktop/phd_courses/XEP33SAM/data/oxford_105k/imgdesc105k.dat');
fid = fopen('/home/marik0/Desktop/phd_courses/XEP33SAM/data/oxford_105k/CNNdesc105k.dat');

X = fread(fid, [128, inf], 'single=>single');
fclose(fid);
Names = textread('/home/marik0/Desktop/phd_courses/XEP33SAM/data/oxford_105k/imagenames105k.txt', '%s');

k = 6;
query_id = 512;

testset = X(:, query_id);
flann_set_distance_type('euclidean');

if (exist('flann_index.mat', 'file') == 2)
    
    fprintf('File exists, loading index!\n');
    index = flann_load_index('flann_index.mat', X);
    pp = load('flann_params.mat');
    parameters = pp.parameters;
else
    % Parameters discovered via the autotune method for precision equal to 0.9
    build_params.algorithm = 'kmeans';
    build_params.target_precision = 0.9;
    build_params.build_weight = 0.01;
    build_params.memory_weight = 0;
    build_params.checks = 960;
    build_params.branching = 32;
    build_params.centers_init = 'random';
    build_params.trees = 4;
    build_params.leaf_max_size = 4;
    build_params.iterations = 5;

    t0 = cputime;
    [index, parameters] = flann_build_index(X, build_params);
    flann_idx = cputime - t0;
    
    % Save index and params to save time
    flann_save_index(index, 'flann_index.mat');
    save('flann_params.mat', 'parameters');
    
    fprintf ('Flann index building  = %.3f s\n', flann_idx);
end

t0 = cputime;
[result, dists] = flann_search(index,testset,k,parameters);
flann_srch = cputime - t0;

flann_free_index(index);

for idx = 1:k
    i = result(idx);
    name = char(Names(i));
    A = imread(strcat('/home/marik0/Desktop/phd_courses/XEP33SAM/data/oxford_105k/oxc-complete/', name, '.jpg'));
    subplot(1, k, idx);
    imshow(A);
end

 
 fprintf ('Flann query time = %.3f s\n', flann_srch);
 fprintf ('Flann sum of distances = %3f \n', sum(dists));
