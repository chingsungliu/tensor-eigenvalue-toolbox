function [x, lambda, res, nit, innit, hal, history] = HONI_with_history( ...
    AA, m, tol, linear_solver, initial_vector, maxit)
%HONI_WITH_HISTORY  Port-parity variant of HONI.m with full per-outer-iter history.
%
%  Simplifications vs HONI.m:
%    - No varargin/varargout; explicit parameters + fixed outputs
%    - AA must already be unfolding (caller does ten2mat if tensor input)
%    - initial_vector must be provided (no default rand)
%    - No plot_res support
%
%  All numerical operations are IDENTICAL to HONI.m; only added:
%    - per-outer-iter recording into 8 history arrays (packed into `history` struct)
%    - truncation to 1:nit at the end
%
%  History slot convention (MATLAB 1-based):
%    slot 1    = initial state (line 35-43 of HONI.m)
%    slot k>=2 = state after the (k-1)-th outer iter completes
%  Python port uses 0-based: python_slot = matlab_slot - 1.
%
%  history struct fields:
%    x_history              (n, nit)   eigenvector after each outer iter
%    lambda_history         (nit, 1)   lambda_U after each outer iter
%    y_history              (n, nit)   Multi's output (pre-normalize)  [:,1] = NaN
%    inner_tol_history      (nit, 1)   inner_tol passed to Multi       (1)   = NaN
%    chit_history           (nit, 1)   Multi's nit per outer iter (MATLAB 1-based)
%    hal_per_outer_history  (nit, 1)   sum(hal_inn) per outer iter
%    innit_history          (nit, 1)   cumulative innit
%    hal_accum_history      (nit, 1)   cumulative hal
%
%  Uses local copies of tpv / tenpow / sp_tendiag from HONI.m.

    n = length(initial_vector);

    % --- 初始化 (HONI.m line 35-44) ---
    x         = initial_vector;
    x         = x / norm(x);
    temp      = tpv(AA, x, m) ./ (x.^(m-1));
    lambda_U  = max(temp);
    II        = sp_tendiag( ones(n, 1), m );
    res       = ones(maxit, 1);
    nit       = 1;
    res(1)    = abs( max(temp) - min(temp) ) / lambda_U;
    hal       = 0;
    innit     = 0;

    % --- history buffers (pre-alloc, truncate at end) ---
    x_history             = zeros(n, maxit);
    x_history(:, 1)       = x;
    lambda_history        = nan(maxit, 1);
    lambda_history(1)     = lambda_U;
    y_history             = nan(n, maxit);    % y_history(:, 1) stays NaN
    inner_tol_history     = nan(maxit, 1);    % inner_tol_history(1) stays NaN
    chit_history          = zeros(maxit, 1);  % chit_history(1) = 0
    hal_per_outer_history = zeros(maxit, 1);
    innit_history         = zeros(maxit, 1);
    hal_accum_history     = zeros(maxit, 1);

    % --- 外層 eigenvalue iteration (HONI.m line 46-79) ---
    while  min(res) > tol && nit < maxit

        nit = nit + 1;

        if strcmp(linear_solver, 'exact')
            %%% exact branch (HONI.m line 49-61)
            inner_tol = 1e-10;
            [y, chit, hal_inn]  = Multi(lambda_U*II - AA, x, m, inner_tol);
            hal     = hal + sum(hal_inn);
            innit   = innit + chit;
            temp    = x ./ y;
            lambda_U   = lambda_U - min(temp)^(m-1);
            res(nit,1) = abs( max(temp)^(m-1) - min(temp)^(m-1) ) / lambda_U;
            x       = y / norm(y);

        elseif strcmp(linear_solver, 'inexact')
            %%% inexact branch (HONI.m line 63-75)
            inner_tol = max(1e-10, min(res) * min(x)^(m-1) / nit);
            [y, chit, hal_inn]  = Multi(lambda_U*II - AA, x, m, inner_tol);
            hal     = hal + sum(hal_inn);
            innit   = innit + chit;
            x       = y / norm(y);
            temp    = tpv(AA, x, m) ./ (x.^(m-1));
            lambda_U   = max( temp );
            res(nit,1) = abs( max(temp) - min(temp) ) / lambda_U;

        else
            error('linear_solver must be ''exact'' or ''inexact'', got %s', linear_solver);
        end

        %%% record per-outer-iter history
        x_history(:, nit)            = x;
        lambda_history(nit)          = lambda_U;
        y_history(:, nit)            = y;
        inner_tol_history(nit)       = inner_tol;
        chit_history(nit)            = chit;
        hal_per_outer_history(nit)   = sum(hal_inn);
        innit_history(nit)           = innit;
        hal_accum_history(nit)       = hal;
    end

    lambda = lambda_U;

    % --- truncate to the number of slots actually used (1:nit) ---
    res                    = res(1:nit);
    history.x_history             = x_history(:, 1:nit);
    history.lambda_history        = lambda_history(1:nit);
    history.y_history             = y_history(:, 1:nit);
    history.inner_tol_history     = inner_tol_history(1:nit);
    history.chit_history          = chit_history(1:nit);
    history.hal_per_outer_history = hal_per_outer_history(1:nit);
    history.innit_history         = innit_history(1:nit);
    history.hal_accum_history     = hal_accum_history(1:nit);
end


%%% === local functions (verbatim from HONI.m lines 123-149) ===

function y = tpv( AA, x, m )
% Compute the m-tensor product with vector : A x^(m-1)
    x_m = tenpow(x, m-1);
    y   = AA*x_m;
end

function x_p = tenpow( x, p )
% Compute x^(p) = x ⊗ x ⊗ ... ⊗ x (p-times kronecker product)
    if p == 0
        x_p = 1;
    else
        x_p = x;
        for i = 1:p-1
            x_p = kron(x, x_p);
        end
    end
end

function D = sp_tendiag( d, m )
% Construct m-order n-dim diagonal tensor (mode-1 unfolding) with diag entries d.
    n    = length(d);
    D    = sparse(n^m, 1);
    S    = linspace(1, n^m, n);
    D(S) = d;
    D    = reshape(D, n, n^(m-1));
end
