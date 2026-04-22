function generate_honi_reference()
%GENERATE_HONI_REFERENCE  MATLAB reference .mat files for Python HONI port.
%
%  Writes two .mat files in matlab_ref/hni/:
%    - honi_reference_exact.mat    (linear_solver = 'exact')
%    - honi_reference_inexact.mat  (linear_solver = 'inexact')
%
%  Each .mat contains:
%    inputs:  AA, m, tol, linear_solver, initial_vector, maxit
%    outputs: x, lambda, res, nit, innit, hal
%    history: x_history, lambda_history, y_history, inner_tol_history,
%             chit_history, hal_per_outer_history, innit_history, hal_accum_history
%
%  Test case (Q4 decision, reuse Multi Q5 params):
%    rng(42), n=20, m=3, d ∈ [1, 11], perturbation scale 0.01, density 0.02
%    initial_vector = abs(rand(n,1)) + 0.1 (= Multi Q5 的 b 重命名)
%    tol = 1e-12 (HONI default), maxit = 200
%
%  Both branches share the same AA and initial_vector; only linear_solver differs,
%  so parity tests can isolate the branch-specific logic.
%
%  Uses HONI_with_history (local copy of HONI.m with per-outer-iter history).
%  Original HONI.m is NOT modified.

    scriptdir = fileparts(mfilename('fullpath'));

    %% ==========================================================================
    %% Build shared inputs: AA + initial_vector (Multi Q5 params)
    %% ==========================================================================
    rng(42);
    n              = 20;
    m              = 3;
    tol            = 1e-12;
    maxit          = 200;

    d              = rand(n, 1) * 10 + 1;                  % [1, 11]
    perturbation   = sprand(n, n^(m-1), 0.02) * 0.01;      % tiny, 2% density
    AA             = sp_tendiag(d, m) + perturbation;
    initial_vector = abs(rand(n, 1)) + 0.1;                % positive, away from 0

    fprintf('====================  Shared inputs (Multi Q5 params)  ====================\n');
    fprintf('  n   = %d,  m = %d,  tol = %.0e,  maxit = %d\n', n, m, tol, maxit);
    fprintf('  nnz(AA) = %d,  size(AA) = [%d, %d]\n', nnz(AA), size(AA,1), size(AA,2));
    fprintf('  min(init) = %.6f,  max(init) = %.6f\n\n', ...
            min(initial_vector), max(initial_vector));

    %% ==========================================================================
    %% Case 1: linear_solver = 'exact'
    %% ==========================================================================
    linear_solver = 'exact';
    fprintf('====================  Case 1: linear_solver = ''exact''  ====================\n\n');

    [x, lambda, res, nit, innit, hal, history] = HONI_with_history( ...
        AA, m, tol, linear_solver, initial_vector, maxit);

    % Unpack struct for flat save (Python friendlier)
    x_history             = history.x_history;
    lambda_history        = history.lambda_history;
    y_history             = history.y_history;
    inner_tol_history     = history.inner_tol_history;
    chit_history          = history.chit_history;
    hal_per_outer_history = history.hal_per_outer_history;
    innit_history         = history.innit_history;
    hal_accum_history     = history.hal_accum_history;

    outpath = fullfile(scriptdir, 'honi_reference_exact.mat');
    save(outpath, 'AA', 'm', 'tol', 'linear_solver', 'initial_vector', 'maxit', ...
         'x', 'lambda', 'res', 'nit', 'innit', 'hal', ...
         'x_history', 'lambda_history', 'y_history', 'inner_tol_history', ...
         'chit_history', 'hal_per_outer_history', 'innit_history', 'hal_accum_history');
    fprintf('honi_reference_exact.mat saved: %s\n\n', outpath);
    print_case_report('exact', nit, innit, hal, lambda, res, x, ...
                      chit_history, hal_per_outer_history, ...
                      x_history, lambda_history, y_history);

    %% ==========================================================================
    %% Case 2: linear_solver = 'inexact'
    %% ==========================================================================
    linear_solver = 'inexact';
    fprintf('\n====================  Case 2: linear_solver = ''inexact''  ====================\n\n');

    [x, lambda, res, nit, innit, hal, history] = HONI_with_history( ...
        AA, m, tol, linear_solver, initial_vector, maxit);

    x_history             = history.x_history;
    lambda_history        = history.lambda_history;
    y_history             = history.y_history;
    inner_tol_history     = history.inner_tol_history;
    chit_history          = history.chit_history;
    hal_per_outer_history = history.hal_per_outer_history;
    innit_history         = history.innit_history;
    hal_accum_history     = history.hal_accum_history;

    outpath_b = fullfile(scriptdir, 'honi_reference_inexact.mat');
    save(outpath_b, 'AA', 'm', 'tol', 'linear_solver', 'initial_vector', 'maxit', ...
         'x', 'lambda', 'res', 'nit', 'innit', 'hal', ...
         'x_history', 'lambda_history', 'y_history', 'inner_tol_history', ...
         'chit_history', 'hal_per_outer_history', 'innit_history', 'hal_accum_history');
    fprintf('honi_reference_inexact.mat saved: %s\n\n', outpath_b);
    print_case_report('inexact', nit, innit, hal, lambda, res, x, ...
                      chit_history, hal_per_outer_history, ...
                      x_history, lambda_history, y_history);

end


%%% ============================================================================
%%% local helpers
%%% ============================================================================

function print_case_report(label, nit, innit, hal, lambda, res, x, ...
                           chit_history, hal_per_outer_history, ...
                           x_history, lambda_history, y_history) %#ok<INUSD>
    fprintf('--- %s outputs ---\n', label);
    fprintf('  nit          = %d   (MATLAB 1-based; Python nit_py = nit - 1 = %d)\n', ...
            nit, nit - 1);
    fprintf('  innit        = %d   (total Multi Newton iters, MATLAB 1-based)\n', innit);
    fprintf('  hal          = %d   (total halvings across all Multi calls)\n', hal);
    fprintf('  lambda       = %.15e\n', lambda);
    fprintf('  final res    = %.3e\n', res(end));
    fprintf('  min(x)       = %.6f   (must be > 0 for positive eigenvector)\n', min(x));
    fprintf('  ||x||        = %.15f\n', norm(x));
    fprintf('\n--- %s per-outer-iter stats ---\n', label);
    fprintf('  chit per outer: ');
    for k = 1:nit
        fprintf('%d ', chit_history(k));
    end
    fprintf('\n  hal per outer: ');
    for k = 1:nit
        fprintf('%d ', hal_per_outer_history(k));
    end
    fprintf('\n\n--- %s history shapes ---\n', label);
    fprintf('  x_history      = [%d x %d]\n', size(x_history, 1), size(x_history, 2));
    fprintf('  lambda_history = [%d x 1]\n', length(lambda_history));
    fprintf('  y_history      = [%d x %d]\n', size(y_history, 1), size(y_history, 2));
    fprintf('\n--- %s first & last lambda ---\n', label);
    fprintf('  lambda_history(1)   = %.15e\n', lambda_history(1));
    fprintf('  lambda_history(end) = %.15e\n', lambda_history(end));
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
