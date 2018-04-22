fid = fopen('/home/marik0/Desktop/phd_courses/XEP33SAM/data/oxford_105k/imgdesc105k.dat');
%fid = fopen('/home/marik0/Desktop/phd_courses/XEP33SAM/data/oxford_105k/CNNdesc105k.dat');

X = fread(fid, [128, inf], 'single=>single');
fclose(fid);
Names = textread('/home/marik0/Desktop/phd_courses/XEP33SAM/data/oxford_105k/imagenames105k.txt', '%s');

index = randperm(104933);
Xtrain = X(:, index(1:103933));
Xquery = X(:, index(103934:104933)); 

k = 100;
nquery = 1000;

%----------------------- PQ ---------------------------
nsq = 2;
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
fprintf('ADC sum of distances = %.3f (m=8)\n', sum(sum(dis_pqc)));


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

i = 100;
%j = 1;
r_at_i(j) = length (find (nn_ranks_pqc <= i & nn_ranks_pqc <= k)) / nquery * 100;
fprintf ('ADC r@%3d = %.3f (m=4, k*=8)\n', i, r_at_i(j)); 
j = j+1;

%r100_m4 = [4.0;25.8;42.9;62.6;11.5];
%r100_m8 = [19.5;50.0;70.6;85.0;13.6];
%r100_m2 = [1.3;5.6;20.4;39.4;6.1];

x = [4 16 32 256]
r100_m4 = [4.0;25.8;42.9;62.6];
r100_m8 = [19.5;50.0;70.6;85.0];
r100_m2 = [1.3;5.6;20.4;39.4];

plot(x, r100_m2, '-*')
plot(x, r100_m4, '-x')
plot(x, r100_m8, '-^')
legend({'ADC m=2', 'ADC m=4', 'ADC m=8'}, 'Location', 'southeast');

xticks(x);
xlabel('code length (bits)');
ylabel('R@100');
