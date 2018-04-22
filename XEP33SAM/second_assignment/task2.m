% Select 1000 random indices as test set
% Train each algo using the rest of the data
% Time database and tree build
% Time query
% Report individual and totals

fid = fopen('/home/marik0/Desktop/phd_courses/XEP33SAM/data/oxford_105k/imgdesc105k.dat');
%fid = fopen('/home/marik0/Desktop/phd_courses/XEP33SAM/data/oxford_105k/CNNdesc105k.dat');

X = fread(fid, [128, inf], 'single=>single');
fclose(fid);
Names = textread('/home/marik0/Desktop/phd_courses/XEP33SAM/data/oxford_105k/imagenames105k.txt', '%s');

index = randperm(104933);
Xtrain = X(:, index(1:103933));
Xquery = X(:, index(103934:104933)); 

k = 1000;
nquery = 1000;

%----------------------- PQ ---------------------------
nsq = 4;
% Learn the PQ code structure
t0 = cputime;
pq = pq_new (nsq, Xtrain)
tpqlearn = cputime - t0;
   
% encode the database vectors
t1 = cputime;
cbase = pq_assign (pq, Xtrain);
tpqencode = cputime - t1;

% Print this only if we perform the calculations
fprintf ('ADC learn  = %.3f s (m=4)\n', tpqlearn);
fprintf ('ADC encode = %.3f s (m=4)\n', tpqencode);

t2 = cputime;
[ids_pqc, dis_pqc] = pq_search (pq, cbase, Xquery, k);
tpq = cputime - t2;

fprintf ('ADC search = %.3f s  for %d query vectors in a database of %d vectors\n', tpq, 1000, 105000);
fprintf('ADC sum of distances = %.3f (m=4)\n', sum(sum(dis_pqc)));

%----------------- FLANN ----------------------------------------
flann_set_distance_type('euclidean');

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

t3 = cputime;
[index, parameters] = flann_build_index(Xtrain, build_params);
flann_idx_t = cputime - t3;
     
fprintf ('Flann index building  = %.3f s\n', flann_idx_t);

t4 = cputime;
[ids_fl, dis_fl] = flann_search(index, Xquery, k, parameters);
flann_search_t = cputime - t4;

fprintf ('Flann query time = %.3f s\n', flann_search_t);
fprintf ('Flann sum of distances = %3f \n', sum(sum(dis_fl)));

flann_free_index(index);

%--------------------- Exact kNN ----------------------------
t5 = cputime;
[ids_gnd, dist_knn] = knnsearch(transpose(Xtrain), transpose(Xquery), 'K', k, 'NSMethod', 'exhaustive');
knn_exact_t = cputime - t5;

fprintf ('Exact kNN search time = %.3f s\n', knn_exact_t);
fprintf ('Exact kNN sum of distances = %3f \n', sum(sum((dist_knn).^2)));


%---[ Compute search statistics for ADC]---
nn_ranks_pqc = zeros (nquery, 1);
r_at_i = zeros(10, 1);
hist_pqc = zeros (k+1, 1);
for i = 1:nquery
  gnd_ids = ids_gnd(i);
  
    nn_pos = find (ids_pqc(i, :) == gnd_ids);
    
    if length (nn_pos) == 1
      nn_ranks_pqc (i) = nn_pos;
    else
      nn_ranks_pqc (i) = k + 1; 
    end
end
nn_ranks_pqc = sort (nn_ranks_pqc);

j = 1;
for i = [1 2 5 10 20 50 100 200 500 1000]
  if i <= k
    r_at_i(j) = length (find (nn_ranks_pqc <= i & nn_ranks_pqc <= k)) / nquery * 100;
    fprintf ('ADC r@%3d = %.3f (m=4)\n', i, r_at_i(j)); 
    j = j+1;
  end
end

%---[ Compute search statistics for FLANN]---
ids_fl = transpose(ids_fl);
nn_ranks_fl = zeros (nquery, 1);
hist_fl = zeros (k+1, 1);
r_at_i_fl = zeros(10, 1);

for i = 1:nquery
  gnd_ids = ids_gnd(i);
  
    nn_pos = find (ids_fl(i, :) == gnd_ids);
    
    if length (nn_pos) == 1
      nn_ranks_fl (i) = nn_pos;
    else
      nn_ranks_fl (i) = k + 1; 
    end
end
nn_ranks_fl = sort (nn_ranks_fl);

j = 1;
for i = [1 2 5 10 20 50 100 200 500 1000]
  if i <= k
    r_at_i_fl(j) = length (find (nn_ranks_fl <= i & nn_ranks_fl <= k)) / nquery * 100;
    fprintf ('FLANN r@%3d = %.3f\n', i, r_at_i_fl(j));
    j = j+1;
  end
end

x = [1 2 5 10 20 50 100 200 500 1000];
plot(x, r_at_i_fl);
hold on;
plot(x, r_at_i, '-^');
hold on;
%legend({'ADC m=8', 'FLANN'}, 'Location', 'southeast');
%hold on;


%----------------------- PQ ---------------------------
nsq = 8;
% Learn the PQ code structure
t0 = cputime;
pq = pq_new (nsq, Xtrain)
tpqlearn = cputime - t0;
   
% encode the database vectors
t1 = cputime;
cbase = pq_assign (pq, Xtrain);
tpqencode = cputime - t1;

% Print this only if we perform the calculations
fprintf ('ADC learn  = %.3f s (m=8)\n', tpqlearn);
fprintf ('ADC encode = %.3f s (m=8)\n', tpqencode);

t2 = cputime;
[ids_pqc, dis_pqc] = pq_search (pq, cbase, Xquery, k);
tpq = cputime - t2;

fprintf ('ADC search = %.3f s  for %d query vectors in a database of %d vectors\n', tpq, 1000, 105000);
fprintf('ADC sum of distances = %.3f (m=8) \n', sum(sum(dis_pqc)));

%---[ Compute search statistics for ADC]---
nn_ranks_pqc = zeros (nquery, 1);
r_at_i = zeros(10, 1);
hist_pqc = zeros (k+1, 1);
for i = 1:nquery
  gnd_ids = ids_gnd(i);
  
    nn_pos = find (ids_pqc(i, :) == gnd_ids);
    
    if length (nn_pos) == 1
      nn_ranks_pqc (i) = nn_pos;
    else
      nn_ranks_pqc (i) = k + 1; 
    end
end
nn_ranks_pqc = sort (nn_ranks_pqc);

j = 1;
for i = [1 2 5 10 20 50 100 200 500 1000]
  if i <= k
    r_at_i(j) = length (find (nn_ranks_pqc <= i & nn_ranks_pqc <= k)) / nquery * 100;
    fprintf ('ADC r@%3d = %.3f (m=8)\n', i, r_at_i(j)); 
    j = j+1;
  end
end

plot(x, r_at_i, '-o');
hold on;
%legend({'ADC m=8', 'FLANN', 'ADC m=4'}, 'Location', 'southeast');


%----------------------- PQ ---------------------------
nsq = 16;
% Learn the PQ code structure
t0 = cputime;
pq = pq_new (nsq, Xtrain)
tpqlearn = cputime - t0;
   
% encode the database vectors
t1 = cputime;
cbase = pq_assign (pq, Xtrain);
tpqencode = cputime - t1;

% Print this only if we perform the calculations
fprintf ('ADC learn  = %.3f s (m=16)\n', tpqlearn);
fprintf ('ADC encode = %.3f s (m=16)\n', tpqencode);

t2 = cputime;
[ids_pqc, dis_pqc] = pq_search (pq, cbase, Xquery, k);
tpq = cputime - t2;

fprintf ('ADC search = %.3f s  for %d query vectors in a database of %d vectors\n', tpq, 1000, 105000);
fprintf('ADC sum of distances = %.3f (m=16) \n', sum(sum(dis_pqc)));

%---[ Compute search statistics for ADC]---
nn_ranks_pqc = zeros (nquery, 1);
r_at_i = zeros(10, 1);
hist_pqc = zeros (k+1, 1);
for i = 1:nquery
  gnd_ids = ids_gnd(i);
  
    nn_pos = find (ids_pqc(i, :) == gnd_ids);
    
    if length (nn_pos) == 1
      nn_ranks_pqc (i) = nn_pos;
    else
      nn_ranks_pqc (i) = k + 1; 
    end
end
nn_ranks_pqc = sort (nn_ranks_pqc);

j = 1;
for i = [1 2 5 10 20 50 100 200 500 1000]
  if i <= k
    r_at_i(j) = length (find (nn_ranks_pqc <= i & nn_ranks_pqc <= k)) / nquery * 100;
    fprintf ('ADC r@%3d = %.3f (m=16)\n', i, r_at_i(j)); 
    j = j+1;
  end
end

plot(x, r_at_i, '-+');
hold on;
legend({'FLANN', 'ADC m=4', 'ADC m=8', 'ADC m=16'}, 'Location', 'southeast');



