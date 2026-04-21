function generate_layer2_reference()
%GENERATE_LAYER2_REFERENCE  MATLAB reference .mat files for Phase-D layer-2
%  Python port parity tests. Writes two .mat files:
%    - ten2mat_reference.mat     (for python/test_ten2mat_parity.py)
%    - sp_Jaco_Ax_reference.mat  (for python/test_sp_Jaco_Ax_parity.py)
%
%  ten2mat / idx_create / sp_Jaco_Ax / tenpow are verbatim copies of the
%  nested functions in matlab_ref/hni/HONI.m (ten2mat:152-177) and
%  matlab_ref/hni/Multi.m (sp_Jaco_Ax:79-87; tenpow:66-76).

    rng(42);

    %%% ============================================================
    %%% ten2mat test cases
    %%% ============================================================
    % Case 1: n=3, m=3
    T1 = rand(3, 3, 3);
    k1 = 1;
    B1 = ten2mat(T1, k1);
    n1 = 3; m1 = 3;

    % Case 2: n=4, m=3
    T2 = rand(4, 4, 4);
    k2 = 1;
    B2 = ten2mat(T2, k2);
    n2 = 4; m2 = 3;

    % Case 3: n=2, m=5
    T3 = rand(2, 2, 2, 2, 2);
    k3 = 1;
    B3 = ten2mat(T3, k3);
    n3 = 2; m3 = 5;

    scriptdir = fileparts(mfilename('fullpath'));
    outpath_ten2mat = fullfile(scriptdir, 'ten2mat_reference.mat');
    save(outpath_ten2mat, ...
        'T1','B1','k1','n1','m1', ...
        'T2','B2','k2','n2','m2', ...
        'T3','B3','k3','n3','m3');

    %%% ============================================================
    %%% sp_Jaco_Ax test cases
    %%% ============================================================
    % Case 1: n=3, m=3 (basic)
    nj1 = 3; mj1 = 3;
    AA_j1 = rand(nj1, nj1^(mj1-1));
    xj1   = rand(nj1, 1);
    J1    = sp_Jaco_Ax(AA_j1, xj1, mj1);

    % Case 2: n=4, m=4 (balanced, p=3 iterations)
    nj2 = 4; mj2 = 4;
    AA_j2 = rand(nj2, nj2^(mj2-1));
    xj2   = rand(nj2, 1);
    J2    = sp_Jaco_Ax(AA_j2, xj2, mj2);

    % Case 3: n=2, m=5 (high order, small dim, p=4 iterations)
    nj3 = 2; mj3 = 5;
    AA_j3 = rand(nj3, nj3^(mj3-1));
    xj3   = rand(nj3, 1);
    J3    = sp_Jaco_Ax(AA_j3, xj3, mj3);

    outpath_jaco = fullfile(scriptdir, 'sp_Jaco_Ax_reference.mat');
    save(outpath_jaco, ...
        'AA_j1','xj1','J1','nj1','mj1', ...
        'AA_j2','xj2','J2','nj2','mj2', ...
        'AA_j3','xj3','J3','nj3','mj3');

    %%% ============================================================
    %%% Report
    %%% ============================================================
    fprintf('ten2mat_reference.mat saved:    %s\n', outpath_ten2mat);
    fprintf('sp_Jaco_Ax_reference.mat saved: %s\n\n', outpath_jaco);

    fprintf('--- ten2mat sanity values ---\n');
    fprintf('case1 (n=3,m=3,k=1): size(B)=[%d,%d], B(1,1)=%.15f, B(1,end)=%.15f\n', ...
        size(B1,1), size(B1,2), B1(1,1), B1(1,end));
    fprintf('case2 (n=4,m=3,k=1): size(B)=[%d,%d], B(1,1)=%.15f, B(4,end)=%.15f\n', ...
        size(B2,1), size(B2,2), B2(1,1), B2(4,end));
    fprintf('case3 (n=2,m=5,k=1): size(B)=[%d,%d], B(1,1)=%.15f, B(2,end)=%.15f\n\n', ...
        size(B3,1), size(B3,2), B3(1,1), B3(2,end));

    fprintf('--- sp_Jaco_Ax sanity values ---\n');
    % sparse indexed values → wrap in full() for fprintf
    fprintf('case1 (n=3,m=3): size(J)=[%d,%d], nnz=%d, J(1,1)=%.15f\n', ...
        size(J1,1), size(J1,2), nnz(J1), full(J1(1,1)));
    fprintf('case2 (n=4,m=4): size(J)=[%d,%d], nnz=%d, J(1,1)=%.15f\n', ...
        size(J2,1), size(J2,2), nnz(J2), full(J2(1,1)));
    fprintf('case3 (n=2,m=5): size(J)=[%d,%d], nnz=%d, J(1,1)=%.15f\n', ...
        size(J3,1), size(J3,2), nnz(J3), full(J3(1,1)));
end


%%% === local functions (verbatim from HONI.m and Multi.m) ===

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

function J = sp_Jaco_Ax(AA, x, m)
% here return a matrix F'(x) where F(x)=Ax^(m-1)
    I = speye(length(x));
    J = 0;
    p = m-1;
    for i = 1:p
        J = J + AA*kron( tenpow(x,i-1) , kron( I , tenpow(x,p-i) ) );
    end
end

function x_p = tenpow(x, p)
% Compute x^(p) = x@x@...@x
    if p == 0
        x_p = 1;
    else
        x_p = x;
        for i = 1:p-1
            x_p = kron(x, x_p);
        end
    end
end
