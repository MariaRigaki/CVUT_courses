k = 6;  % number of elements to be returned
query_id = 513;
nsq = 8;  % number of subquantizers to be used (m in the paper)
% coarsek = 256;


%fid = fopen('/home/marik0/Desktop/phd_courses/XEP33SAM/data/oxford_105k/imgdesc105k.dat');
fid = fopen('/home/marik0/Desktop/phd_courses/XEP33SAM/data/oxford_105k/CNNdesc105k.dat');

X = fread(fid, [128, inf], 'single=>single');
fclose(fid);


fbase = X;
fquery = X(:, query_id);
Names = textread('/home/marik0/Desktop/phd_courses/XEP33SAM/data/oxford_105k/imagenames105k.txt', '%s');

if (exist('pq_db.mat', 'file') == 2) && (exist('pq.mat', 'file') == 2)
    
    fprintf('Files exist, loading them!\n');
    cbase_l = load('pq_db.mat');
    cbase = cbase_l.cbase;
    pq_l = load('pq.mat');
    pq = pq_l.pq;
    
else
    fprintf('Files do not exist, calculating for all 105k images!\n');
    
    % Learn the PQ code structure
    t0 = cputime;
    pq = pq_new (nsq, X)
    tpqlearn = cputime - t0;
   
    % encode the database vectors
    t0 = cputime;
    cbase = pq_assign (pq, X);
    tpqencode = cputime - t0;

    % Save the objects so we don't have to do it again
    save('pq.mat', 'pq');
    save('pq_db.mat', 'cbase');
    
    % Print this only if we perform the calculations
    fprintf ('ADC learn  = %.3f s\n', tpqlearn);
    fprintf ('ADC encode = %.3f s\n', tpqencode);
end
%---[ perform the search and compare with the ground-truth ]---
t0 = cputime;
[ids_pqc, dis_pqc] = pq_search (pq, cbase, fquery, k);
tpq = cputime - t0;

fprintf ('ADC search = %.3f s  for %d query vectors in a database of %d vectors\n', tpq, 1, 105000);

for idx = 1:k
    i = ids_pqc(idx);
    name = char(Names(i));
    A = imread(strcat('/home/marik0/Desktop/phd_courses/XEP33SAM/data/oxford_105k/oxc-complete/', name, '.jpg'));
    subplot(1, k, idx);
    imshow(A);
end
fprintf ('PQ sum of distances = %3f \n', sum(dis_pqc));

[ids_gnd, dist_gnd] = knnsearch(transpose(X), transpose(X(:, query_id)), 'K', k, 'NSMethod', 'exhaustive');
%numquery = k;
%pq_test_compute_stats;
