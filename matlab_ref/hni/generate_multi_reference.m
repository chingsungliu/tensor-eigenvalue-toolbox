function generate_multi_reference()
%GENERATE_MULTI_REFERENCE  MATLAB reference .mat file for Python Multi port.
%
%  Writes matlab_ref/hni/multi_reference.mat with:
%    inputs:  AA, b, m, tol
%    outputs: u, nit, hal
%    history: u_history, res_history, theta_history, hal_history, v_history
%
%  Q5 test case (hazard analysis §六):
%    rng(42), n=20, m=3, d ∈ [1, 11], perturbation scale 0.01, density 0.02.
%    Diagonally dominant M-tensor; halving path not exercised by design.
%    See memory/feedback_multi_halving_fragility.md for why halving-path
%    parity is deferred to HONI integration tests.
%
%  Calls Multi_with_history (local copy of Multi.m with history output). The
%  original Multi.m is NOT modified. Both files live in matlab_ref/hni/ —
%  cd here or addpath before running.

    scriptdir = fileparts(mfilename('fullpath'));

    fprintf('====================  Q5: standard case  ====================\n\n');

    rng(42);
    n   = 20;
    m   = 3;
    tol = 1e-10;

    d            = rand(n, 1) * 10 + 1;                  % [1, 11]
    perturbation = sprand(n, n^(m-1), 0.02) * 0.01;      % tiny, 2% density
    AA           = sp_tendiag(d, m) + perturbation;
    b            = abs(rand(n, 1)) + 0.1;

    [u, nit, hal, u_history, res_history, theta_history, hal_history, v_history] = ...
        Multi_with_history(AA, b, m, tol);

    outpath = fullfile(scriptdir, 'multi_reference.mat');
    save(outpath, 'AA', 'b', 'm', 'tol', 'u', 'nit', 'hal', ...
         'u_history', 'res_history', 'theta_history', 'hal_history', 'v_history');
    fprintf('multi_reference.mat saved: %s\n\n', outpath);
    print_case_report('Q5', n, m, tol, AA, b, u, nit, hal, ...
                      u_history, res_history, theta_history, hal_history, v_history);
end


%%% ============================================================================
%%% local helpers
%%% ============================================================================

function print_case_report(label, n, m, tol, AA, b, u, nit, hal, ...
                           u_history, res_history, theta_history, hal_history, v_history) %#ok<INUSD>
%PRINT_CASE_REPORT  Uniform console report for a Multi reference case.
    fprintf('--- %s inputs ---\n', label);
    fprintf('  n   = %d\n', n);
    fprintf('  m   = %d\n', m);
    fprintf('  tol = %.0e\n', tol);
    fprintf('  nnz(AA) = %d   size(AA) = [%d, %d]\n', nnz(AA), size(AA, 1), size(AA, 2));
    fprintf('  min(b)  = %.6f,  max(b) = %.6f\n', min(b), max(b));
    fprintf('\n--- %s outputs ---\n', label);
    fprintf('  nit = %d   (MATLAB 1-based; Python nit_py = nit - 1 = %d)\n', nit, nit - 1);
    fprintf('  final residual  = %.3e\n', res_history(end));
    fprintf('  min(u)          = %.6f   (must be > 0 for positive-solution invariant)\n', min(u));
    fprintf('  total halvings  = %d\n', sum(hal_history));
    fprintf('\n--- %s history shapes ---\n', label);
    fprintf('  u_history     = [%d x %d]\n', size(u_history, 1), size(u_history, 2));
    fprintf('  res_history   = [%d x 1]\n', length(res_history));
    fprintf('  theta_history = [%d x 1]\n', length(theta_history));
    fprintf('  hal_history   = [%d x 1]\n', length(hal_history));
    fprintf('  v_history     = [%d x %d]\n', size(v_history, 1), size(v_history, 2));
    fprintf('\n--- %s first & last res ---\n', label);
    fprintf('  res_history(1)   = %.15e\n', res_history(1));
    fprintf('  res_history(end) = %.15e\n', res_history(end));
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
