function generate_nni_reference()
%GENERATE_NNI_REFERENCE  MATLAB reference .mat file for Python NNI port.
%
%  Writes matlab_ref/nni/nni_reference.mat with:
%    inputs:  AA, m, tol, initial_vector, maxit
%    outputs: x, lambda, lambda_L, res, nit, chit
%    history: x_history, lambda_U_history, lambda_L_history,
%             w_history, y_history, chit_history, hit_per_outer_history
%
%  Test case (Q7 decision, reuse Multi Q5 / HONI Q4 params verbatim):
%    rng(42), n=20, m=3, d ∈ [1, 11], perturbation scale 0.01, density 0.02
%    initial_vector = abs(rand(n,1)) + 0.1
%    tol = 1e-12 (NNI.m default), maxit = 200
%
%  This AA + initial_vector matches Multi Q5 and HONI Q4 reference .mat files
%  exactly — same rng(42), same construction sequence. **Python side must also
%  reproduce the same AA** for parity to match (see test_nni_parity.py).
%
%  Uses NNI_with_history (local copy of canonical NNI.m 'GE' path with history).
%  Original 2020_HNI_Revised/NNI.m is NOT modified.

    scriptdir = fileparts(mfilename('fullpath'));

    rng(42);
    n              = 20;
    m              = 3;
    tol            = 1e-12;
    maxit          = 200;

    d              = rand(n, 1) * 10 + 1;                  % [1, 11]
    perturbation   = sprand(n, n^(m-1), 0.02) * 0.01;      % tiny, 2% density
    AA             = sp_tendiag(d, m) + perturbation;
    initial_vector = abs(rand(n, 1)) + 0.1;                % positive, away from 0

    fprintf('====================  NNI reference (Multi Q5 / HONI Q4 params)  ====================\n');
    fprintf('  n   = %d,  m = %d,  tol = %.0e,  maxit = %d\n', n, m, tol, maxit);
    fprintf('  nnz(AA) = %d,  size(AA) = [%d, %d]\n', nnz(AA), size(AA,1), size(AA,2));
    fprintf('  min(init) = %.6f,  max(init) = %.6f\n\n', ...
            min(initial_vector), max(initial_vector));

    [x, lambda, lambda_L, res, nit, chit, history] = NNI_with_history( ...
        AA, m, tol, initial_vector, maxit);

    % Unpack struct for flat save (Python friendlier)
    x_history              = history.x_history;
    lambda_U_history       = history.lambda_U_history;
    lambda_L_history       = history.lambda_L_history;
    w_history              = history.w_history;
    y_history              = history.y_history;
    chit_history           = history.chit_history;
    hit_per_outer_history  = history.hit_per_outer_history;

    outpath = fullfile(scriptdir, 'nni_reference.mat');
    save(outpath, 'AA', 'm', 'tol', 'initial_vector', 'maxit', ...
         'x', 'lambda', 'lambda_L', 'res', 'nit', 'chit', ...
         'x_history', 'lambda_U_history', 'lambda_L_history', ...
         'w_history', 'y_history', 'chit_history', 'hit_per_outer_history');
    fprintf('nni_reference.mat saved: %s\n\n', outpath);
    print_case_report(nit, chit, lambda, lambda_L, res, x, ...
                      hit_per_outer_history, x_history, lambda_U_history, lambda_L_history);
end


%%% ============================================================================
%%% local helpers
%%% ============================================================================

function print_case_report(nit, chit, lambda, lambda_L, res, x, ...
                           hit_per_outer_history, x_history, ...
                           lambda_U_history, lambda_L_history) %#ok<INUSD>
    fprintf('--- NNI outputs ---\n');
    fprintf('  nit          = %d   (MATLAB 1-based; Python nit_py = nit - 1 = %d)\n', ...
            nit, nit - 1);
    fprintf('  chit         = %d   (canonical NNI.m: halving disabled → always 0)\n', chit);
    fprintf('  lambda (λ_U) = %.15e\n', lambda);
    fprintf('  lambda_L     = %.15e\n', lambda_L);
    fprintf('  spread       = %.3e   (λ_U - λ_L, convergence measure)\n', lambda - lambda_L);
    fprintf('  final res    = %.3e\n', res(end));
    fprintf('  min(x)       = %.6e   (must be > 0 for positive eigenvector)\n', min(x));
    fprintf('  ||x||        = %.15f\n', norm(x));
    fprintf('\n--- NNI per-outer-iter stats (should all be 0, canonical no halving) ---\n');
    fprintf('  hit per outer: ');
    for k = 1:nit
        fprintf('%d ', hit_per_outer_history(k));
    end
    fprintf('\n\n--- NNI history shapes ---\n');
    fprintf('  x_history        = [%d x %d]\n', size(x_history, 1), size(x_history, 2));
    fprintf('  lambda_U_history = [%d x 1]\n', length(lambda_U_history));
    fprintf('  lambda_L_history = [%d x 1]\n', length(lambda_L_history));
    fprintf('\n--- first & last lambda ---\n');
    fprintf('  lambda_U_history(1)   = %.15e   (initial max(temp))\n', lambda_U_history(1));
    fprintf('  lambda_U_history(end) = %.15e\n', lambda_U_history(end));
    fprintf('  lambda_L_history(1)   = %.15e   (initial min(temp))\n', lambda_L_history(1));
    fprintf('  lambda_L_history(end) = %.15e\n', lambda_L_history(end));
end


function D = sp_tendiag( d, m )
% Construct m-order n-dim diagonal tensor (mode-1 unfolding) with diag entries d.
% (verbatim from matlab_ref/hni/HONI.m lines 142-149)
    n    = length(d);
    D    = sparse(n^m, 1);
    S    = linspace(1, n^m, n);
    D(S) = d;
    D    = reshape(D, n, n^(m-1));
end
