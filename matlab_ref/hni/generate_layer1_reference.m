function generate_layer1_reference()
%GENERATE_LAYER1_REFERENCE  Produce MATLAB reference .mat file(s) for the
%  Phase-D layer-1 Python port parity tests.
%
%  Current scope: tpv. Writes tpv_reference.mat (for python/test_tpv_parity.py).
%  sp_tendiag cases will be added to this script in a later commit.
%
%  The tpv and tenpow functions below are verbatim copies of the nested
%  functions in matlab_ref/hni/HONI.m (tenpow also defined identically in
%  Multi.m). They are local functions here so this script can call them —
%  the canonical sources keep them nested and thus privately scoped.

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

    %%% === Report ===
    fprintf('tpv_reference.mat saved: %s\n\n', outpath_tpv);
    fprintf('--- tpv sanity values ---\n');
    fprintf('case1 (m=3, n=4): y(1) = %.15f\n', y_tpv1(1));
    fprintf('case2 (m=4, n=3): y(1) = %.15f\n', y_tpv2(1));
    fprintf('case3 (m=2, n=5): y(1) = %.15f\n', y_tpv3(1));
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
