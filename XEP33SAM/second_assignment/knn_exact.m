

% Read the descriptor
% Compute exact kNNfunction out_data = knn(test_data,tr_data,k)
% Store the results in the groundtruth file

%fid = fopen('/home/marik0/Desktop/phd_courses/XEP33SAM/data/oxford_105k/imgdesc105k.dat');
fid = fopen('/home/marik0/Desktop/phd_courses/XEP33SAM/data/oxford_105k/CNNdesc105k.dat');

X = fread(fid, [128, inf], 'single=>single');
fclose(fid);
Names = textread('/home/marik0/Desktop/phd_courses/XEP33SAM/data/oxford_105k/imagenames105k.txt', '%s');

k = 6;
query_id = 513;

%t0 = cputime;
%Mdl = ExhaustiveSearcher(transpose(X));
%t1 = cputime - t0;

t0 = cputime;
[neighbors, knn_dists] = knnsearch(transpose(X), transpose(X(:, query_id)), 'K', k, 'NSMethod', 'exhaustive');
t1 = cputime - t0;

for idx = 1:numel(neighbors)
    i = neighbors(idx);
    name = char(Names(i));
    B = imread(strcat('/home/marik0/Desktop/phd_courses/XEP33SAM/data/oxford_105k/oxc-complete/', name, '.jpg'));
    subplot(1, k, idx);
    imshow(B);
end

fprintf ('kNN search time = %.3f s\n', t1);
fprintf ('Exact kNN sum of distances = %3f \n', sum((knn_dists).^2));
