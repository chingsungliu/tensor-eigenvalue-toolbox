function generate_multi_reference()
%GENERATE_MULTI_REFERENCE  MATLAB reference for Python Multi port parity test.
%
%  Writes matlab_ref/hni/multi_reference.mat with:
%    inputs:  AA, b, m, tol
%    outputs: u, nit, hal
%    history: u_history, res_history, theta_history, hal_history, v_history
%
%  Q5 test case (hazard analysis §六, Day 2 decisions):
%    n=20, m=3, diagonally-dominant M-tensor + small sparse perturbation.
%    rng(42) deterministic. Target: halving rarely triggered.
%
%  This script calls Multi_with_history (local copy of Multi.m with history
%  output). The original Multi.m is NOT modified. Both files live in the
%  same directory (matlab_ref/hni/), so either `cd` here or `addpath` before
%  running.

    rng(42);

    n   = 20;
    m   = 3;
    tol = 1e-10;

    % --- (a) 對角 d ∈ [1, 11]，明顯正、保證對角絕對支配 ---
    d = rand(n, 1) * 10 + 1;

    % --- (b) 小 sparse perturbation：2% density、scale 0.01 ---
    perturbation = sprand(n, n^(m-1), 0.02) * 0.01;

    % --- (c) AA = sp_tendiag(d, m) + perturbation ---
    AA = sp_tendiag(d, m) + perturbation;

    % --- (d) b > 0 保證有正解 ---
    b = abs(rand(n, 1)) + 0.1;

    % --- Run Multi with history ---
    [u, nit, hal, u_history, res_history, theta_history, hal_history, v_history] = ...
        Multi_with_history(AA, b, m, tol);

    % --- Save ---
    scriptdir = fileparts(mfilename('fullpath'));
    outpath = fullfile(scriptdir, 'multi_reference.mat');
    save(outpath, ...
        'AA', 'b', 'm', 'tol', ...
        'u', 'nit', 'hal', ...
        'u_history', 'res_history', 'theta_history', 'hal_history', 'v_history');

    % --- Report ---
    fprintf('multi_reference.mat saved: %s\n\n', outpath);
    fprintf('--- inputs ---\n');
    fprintf('  n   = %d\n', n);
    fprintf('  m   = %d\n', m);
    fprintf('  tol = %.0e\n', tol);
    fprintf('  nnz(AA) = %d   size(AA) = [%d, %d]\n', nnz(AA), size(AA,1), size(AA,2));
    fprintf('  min(b)  = %.6f,  max(b) = %.6f\n', min(b), max(b));
    fprintf('\n--- outputs ---\n');
    fprintf('  nit = %d   (MATLAB 1-based; Python nit_py = nit - 1 = %d)\n', nit, nit - 1);
    fprintf('  final residual = %.3e\n', res_history(end));
    fprintf('  min(u) = %.6f  (must be > 0 for positive-solution invariant)\n', min(u));
    fprintf('  total halvings = %d\n', sum(hal_history));
    fprintf('\n--- history shapes ---\n');
    fprintf('  u_history     = [%d x %d]\n', size(u_history, 1), size(u_history, 2));
    fprintf('  res_history   = [%d x 1]\n', length(res_history));
    fprintf('  theta_history = [%d x 1]\n', length(theta_history));
    fprintf('  hal_history   = [%d x 1]\n', length(hal_history));
    fprintf('  v_history     = [%d x %d]\n', size(v_history, 1), size(v_history, 2));
    fprintf('\n--- first & last res ---\n');
    fprintf('  res_history(1)   = %.15e\n', res_history(1));
    fprintf('  res_history(end) = %.15e\n', res_history(end));

end


%%% === local: sp_tendiag (verbatim from HONI.m lines 142-149) ===

function D = sp_tendiag( d, m )
% Construct m-order n-dim diagonal tensor (mode-1 unfolding) with diag entries d.
    n    = length(d);
    D    = sparse(n^m, 1);
    S    = linspace(1, n^m, n);
    D(S) = d;
    D    = reshape(D, n, n^(m-1));
end
