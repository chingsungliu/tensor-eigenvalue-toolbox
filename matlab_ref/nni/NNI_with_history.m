function [x, lambda, lambda_L, res, nit, chit, history] = NNI_with_history( ...
    AA, m, tol, initial_vector, maxit)
%NNI_WITH_HISTORY  Port-parity variant of 2020_HNI_Revised/NNI.m with full
%                  per-outer-iter history output.
%
%  Scope (matches Python port's 'spsolve' branch, per Session 4 決定):
%    - only 'GE' (MATLAB backslash) linear_solver path → Python 'spsolve'
%    - no 'GTH' (未 port、需 geM.m)
%    - no plot_res
%    - step_length verbatim from canonical NNI.m (halving 整塊註解 → hit ≡ 0)
%
%  Differences from canonical 2020_HNI_Revised/NNI.m:
%    - explicit arguments (no varargin)
%    - AA must already be unfolding (caller does ten2mat if tensor)
%    - initial_vector must be provided (no default rand)
%    - no linear_solver kwarg (hardcoded to GE)
%    - records 7 history arrays per outer iter
%    - returns lambda_L as separate output (canonical NNI.m discards it)
%
%  All numerical computations are IDENTICAL to canonical NNI.m's GE path
%  (the original is not touched).
%
%  History slot convention (MATLAB 1-based):
%    slot 1    = initial state (after line 35 of NNI.m)
%    slot k>=2 = state after the (k-1)-th outer iter completes
%  Python port uses 0-based: python_slot = matlab_slot - 1.
%
%  history struct fields:
%    x_history              (n, nit)   normalized x after each outer iter
%    lambda_U_history       (nit, 1)   lambda_U after each outer iter
%    lambda_L_history       (nit, 1)   lambda_L after each outer iter
%    w_history              (n, nit)   RHS x.^(m-1) per iter  [:,1] = NaN
%    y_history              (n, nit)   normalized Newton dir w/norm(w)  [:,1] = NaN
%    chit_history           (nit, 1)   accumulated halving count (≡ 0)
%    hit_per_outer_history  (nit, 1)   halving this iter (≡ 0)
%
%  Uses local copies of tpv / tenpow / sp_Jaco_Ax (verbatim from NNI.m).

    n = length(initial_vector);

    % --- Initialization (NNI.m line 29-41) ---
    x         = initial_vector;
    x         = x/norm(x);
    temp      = tpv(AA, x, m) ./ (x.^(m-1));
    lambda_U  = max(temp);
    lambda_L  = min(temp);

    res       = ones(maxit, 1);
    res(1)    = (lambda_U - lambda_L) / lambda_U;
    nit       = 1;
    chit      = 0;

    % --- history buffers (pre-alloc, truncate at the end) ---
    x_history             = zeros(n, maxit);
    x_history(:, 1)       = x;
    lambda_U_history      = nan(maxit, 1);
    lambda_U_history(1)   = lambda_U;
    lambda_L_history      = nan(maxit, 1);
    lambda_L_history(1)   = lambda_L;
    w_history             = nan(n, maxit);           % slot 1 stays NaN
    y_history             = nan(n, maxit);           % slot 1 stays NaN
    chit_history          = zeros(maxit, 1);
    hit_per_outer_history = zeros(maxit, 1);

    % --- Outer Newton loop (NNI.m line 42-75) ---
    while min(res) > tol && nit < maxit

        nit = nit + 1;

        % Jacobian + shifted matrix (NNI.m line 44, 46)
        B     = sp_Jaco_Ax(AA, x, m);
        w_rhs = x.^(m-1);                                    % RHS, saved for history
        M     = B - (m-1)*lambda_U*diag(x.^(m-2));           % verbatim NNI.m line 46

        % GE (MATLAB backslash) linear solve (NNI.m line 49)
        w     = (-M) \ w_rhs;
        y     = w / norm(w);

        % step_length (no-halving equivalent; NNI.m canonical halving commented)
        [x, lambda_U, lambda_L, hit] = step_length_no_halving(AA, m, x, y);
        chit = chit + hit;

        res(nit) = (lambda_U - lambda_L) / lambda_U;

        % record per-outer-iter history
        x_history(:, nit)             = x;
        lambda_U_history(nit)         = lambda_U;
        lambda_L_history(nit)         = lambda_L;
        w_history(:, nit)             = w_rhs;
        y_history(:, nit)             = y;
        chit_history(nit)             = chit;
        hit_per_outer_history(nit)    = hit;
    end

    lambda = lambda_U;

    % --- truncate to the number of slots actually used (1:nit) ---
    res                              = res(1:nit);
    history.x_history                = x_history(:, 1:nit);
    history.lambda_U_history         = lambda_U_history(1:nit);
    history.lambda_L_history         = lambda_L_history(1:nit);
    history.w_history                = w_history(:, 1:nit);
    history.y_history                = y_history(:, 1:nit);
    history.chit_history             = chit_history(1:nit);
    history.hit_per_outer_history    = hit_per_outer_history(1:nit);
end


%%% === step_length no-halving (NNI.m line 148-189 with halving commented out) ===

function [x_new, lambda_U_new, lambda_L_new, hit] = step_length_no_halving(AA, m, x, y)
% Canonical NNI.m step_length:
%   - theta = 1 (no halving loop — lines 174-186 整塊註解)
%   - x_new = (m-2)*x + 1*y
%   - normalize, recompute temp, lambda_U_new, lambda_L_new
%   - hit always 0 (halving disabled)
    hit          = 0;
    % theta = 1 (declared but not used since halving disabled)
    x_new        = (m-2)*x + 1*y;                  % canonical NNI.m line 169
    x_new        = x_new / norm(x_new);
    temp         = tpv(AA, x_new, m) ./ (x_new.^(m-1));
    lambda_U_new = max(temp);
    lambda_L_new = min(temp);
end


%%% === local functions (verbatim from 2020_HNI_Revised/NNI.m lines 110-146) ===

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

function J = sp_Jaco_Ax( AA, x, m )
% Return F'(x) where F(x) = A x^(m-1)
    I = speye(length(x));
    J = 0;
    p = m-1;
    for i = 1:p
        J = J + AA*kron( tenpow(x,i-1) , kron( I , tenpow(x,p-i) ) );
    end
end
