function generate_layer2_reference()
%GENERATE_LAYER2_REFERENCE  MATLAB reference .mat for Phase-D layer-2
%  Python port parity tests. Writes:
%    - ten2mat_reference.mat  (for python/test_ten2mat_parity.py)
%
%  ten2mat and its helper idx_create are verbatim copies of the nested
%  functions in matlab_ref/hni/HONI.m (lines 152-177).

    rng(42);

    %%% === ten2mat test cases (all with k=1, MATLAB mode-1 unfolding) ===

    % Case 1: n=3, m=3 (square, balanced)
    T1 = rand(3, 3, 3);
    k1 = 1;
    B1 = ten2mat(T1, k1);
    n1 = 3; m1 = 3;

    % Case 2: n=4, m=3 (higher dim, same order)
    T2 = rand(4, 4, 4);
    k2 = 1;
    B2 = ten2mat(T2, k2);
    n2 = 4; m2 = 3;

    % Case 3: n=2, m=5 (high order on small dim — maximizes column-major exposure:
    %   the "remaining modes" are 4 of them, compound trap surface if order is wrong)
    T3 = rand(2, 2, 2, 2, 2);
    k3 = 1;
    B3 = ten2mat(T3, k3);
    n3 = 2; m3 = 5;

    scriptdir = fileparts(mfilename('fullpath'));
    outpath = fullfile(scriptdir, 'ten2mat_reference.mat');
    save(outpath, ...
        'T1','B1','k1','n1','m1', ...
        'T2','B2','k2','n2','m2', ...
        'T3','B3','k3','n3','m3');

    %%% === Report ===
    fprintf('ten2mat_reference.mat saved: %s\n\n', outpath);
    fprintf('--- ten2mat sanity values ---\n');
    fprintf('case1 (n=3,m=3,k=1): size(B)=[%d,%d], B(1,1)=%.15f, B(1,end)=%.15f\n', ...
        size(B1,1), size(B1,2), B1(1,1), B1(1,end));
    fprintf('case2 (n=4,m=3,k=1): size(B)=[%d,%d], B(1,1)=%.15f, B(4,end)=%.15f\n', ...
        size(B2,1), size(B2,2), B2(1,1), B2(4,end));
    fprintf('case3 (n=2,m=5,k=1): size(B)=[%d,%d], B(1,1)=%.15f, B(2,end)=%.15f\n', ...
        size(B3,1), size(B3,2), B3(1,1), B3(2,end));
end


%%% === local functions (verbatim from matlab_ref/hni/HONI.m:152-177) ===

function B = ten2mat(A, k)
% Construct k-type matrix form of a tensor A, k must less than the order of
% tensor A.
    n = size(A, 1);
    m = length(size(A));

    [temp1, temp2] = idx_create(n, k);
    express = ['reshape(','A(',temp1,'i',temp2,')',',',int2str(1),',',int2str(n^(m-1)),');'];

    B = zeros(n, n^(m-1));
    for i = 1:n
        B(i, :) = eval(express);
    end
end

function [temp1, temp2] = idx_create(n, type)
    temp1 = [];
    temp2 = [];
    for i = 1:n
        if i < type
            temp1 = [temp1, ':,'];
        elseif i > type
            temp2 = [temp2, ',:'];
        end
    end
end
