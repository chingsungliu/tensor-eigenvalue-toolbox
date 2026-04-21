function generate_layer1_reference()
%GENERATE_LAYER1_REFERENCE  Produce MATLAB reference .mat files for the
%  Phase-D layer-1 Python port parity tests. Writes two .mat files:
%    - tpv_reference.mat        (for python/test_tpv_parity.py)
%    - sp_tendiag_reference.mat (for python/test_sp_tendiag_parity.py)
%
%  The tpv, tenpow, and sp_tendiag functions below are verbatim copies of
%  the nested functions in matlab_ref/hni/HONI.m (tenpow also defined
%  identically in Multi.m). They are included as local functions so this
%  script can call them — the canonical sources keep them nested and thus
%  privately scoped.

    rng(42);

    %%% === tpv test cases ===
    % Case 1: m=3, n=4
    m_tpv1 = 3; n_tpv1 = 4;
    AA_tpv1 = rand(n_tpv1, n_tpv1^(m_tpv1-1));
    x_tpv1  = rand(n_tpv1, 1);
    y_tpv1  = tpv(AA_tpv1, x_tpv1, m_tpv1);

    % Case 2: m=4, n=3
    m_tpv2 = 4; n_tpv2 = 3;
    AA_tpv2 = rand(n_tpv2, n_tpv2^(m_tpv2-1));
    x_tpv2  = rand(n_tpv2, 1);
    y_tpv2  = tpv(AA_tpv2, x_tpv2, m_tpv2);

    % Case 3: m=2, n=5 (edge: tenpow(x,1) = x, so y = AA*x)
    m_tpv3 = 2; n_tpv3 = 5;
    AA_tpv3 = rand(n_tpv3, n_tpv3^(m_tpv3-1));
    x_tpv3  = rand(n_tpv3, 1);
    y_tpv3  = tpv(AA_tpv3, x_tpv3, m_tpv3);

    scriptdir = fileparts(mfilename('fullpath'));
    outpath_tpv = fullfile(scriptdir, 'tpv_reference.mat');
    save(outpath_tpv, ...
        'AA_tpv1','x_tpv1','y_tpv1','m_tpv1','n_tpv1', ...
        'AA_tpv2','x_tpv2','y_tpv2','m_tpv2','n_tpv2', ...
        'AA_tpv3','x_tpv3','y_tpv3','m_tpv3','n_tpv3');

    %%% === sp_tendiag test cases ===
    % Case 1: n=3, m=3
    m_sp1 = 3;
    d_sp1 = rand(3, 1);
    D_sp1 = sp_tendiag(d_sp1, m_sp1);

    % Case 2: n=5, m=4
    m_sp2 = 4;
    d_sp2 = rand(5, 1);
    D_sp2 = sp_tendiag(d_sp2, m_sp2);

    % Case 3: n=2, m=5 (edge: large m on small n)
    m_sp3 = 5;
    d_sp3 = rand(2, 1);
    D_sp3 = sp_tendiag(d_sp3, m_sp3);

    outpath_sp = fullfile(scriptdir, 'sp_tendiag_reference.mat');
    save(outpath_sp, ...
        'd_sp1','D_sp1','m_sp1', ...
        'd_sp2','D_sp2','m_sp2', ...
        'd_sp3','D_sp3','m_sp3');

    %%% === Report ===
    fprintf('tpv_reference.mat saved:        %s\n', outpath_tpv);
    fprintf('sp_tendiag_reference.mat saved: %s\n\n', outpath_sp);

    fprintf('--- tpv sanity values ---\n');
    fprintf('case1 (m=3, n=4): y(1) = %.15f\n', y_tpv1(1));
    fprintf('case2 (m=4, n=3): y(1) = %.15f\n', y_tpv2(1));
    fprintf('case3 (m=2, n=5): y(1) = %.15f\n\n', y_tpv3(1));

    fprintf('--- sp_tendiag sanity values ---\n');
    % NOTE: sparse indexed values (e.g. D_sp1(1,1)) must be wrapped in full()
    % before fprintf %.15f — fprintf does not accept sparse arguments.
    fprintf('case1 (n=3,m=3): nnz=%d, D(1,1)=%.15f, D(2,5)=%.15f, D(3,9)=%.15f\n', ...
        nnz(D_sp1), full(D_sp1(1,1)), full(D_sp1(2,5)), full(D_sp1(3,9)));
    fprintf('case2 (n=5,m=4): nnz=%d, D(1,1)=%.15f, D(5,125)=%.15f\n', ...
        nnz(D_sp2), full(D_sp2(1,1)), full(D_sp2(5,125)));
    fprintf('case3 (n=2,m=5): nnz=%d, size=[%d,%d]\n', ...
        nnz(D_sp3), size(D_sp3,1), size(D_sp3,2));
end


%%% === local functions (verbatim from matlab_ref/hni/HONI.m) ===

function y = tpv(AA, x, m)
% Compute the m-tensor product with vector : Ax^(m-1)
    x_m = tenpow(x, m-1);
    y   = AA * x_m;
end

function x_p = tenpow(x, p)
% Compute x^(p) = x@x@...@x  (p-times and '@' denote kronecker product)
    if p == 0
        x_p = 1;
    else
        x_p = x;
        for i = 1:p-1
            x_p = kron(x, x_p);
        end
    end
end

function D = sp_tendiag(d, m)
% Construct m-order, n-dimension diagonal tensor with diagonal entrices d.
    n    = length(d);
    D    = sparse(n^m, 1);
    S    = linspace(1, n^m, n);
    D(S) = d;
    D    = reshape(D, n, n^(m-1));
end
