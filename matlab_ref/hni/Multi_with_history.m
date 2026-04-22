function [u, nit, hal, u_history, res_history, theta_history, hal_history, v_history] = Multi_with_history(AA, b, m, tol)
%MULTI_WITH_HISTORY  Verbatim port of Multi.m with per-iteration history output.
%
%  Only difference from matlab_ref/hni/Multi.m:
%    - adds 5 extra output arrays (u/res/theta/hal/v history) for parity test
%    - records u, v, theta after each outer iter; res/hal re-exposed after truncation
%  All numerical computations are IDENTICAL to Multi.m (the original is not touched).
%
%  History slot convention (MATLAB 1-based):
%    slot 1    = initial state (after line 14 in Multi.m)
%    slot k>=2 = state after the (k-1)-th outer Newton iter
%  Python port uses 0-based: python_slot = matlab_slot - 1.
%
%  Uses local copies of tpv / tenpow / sp_Jaco_Ax from Multi.m (lines 60-87).

    n = length(b);

    % --- 初始化 (Multi.m line 14-21) ---
    u      = b/norm(b);
    b      = b.^(m-1);
    na     = sqrt(norm(AA,'inf')*norm(AA,1));
    nb     = norm(b);
    temp   = tpv(AA,u,m);
    res    = (na+nb)*ones(100,1);
    res(1) = norm( temp - b );
    hal    = zeros(100,1);
    nit    = 1;

    % --- history buffers (pre-allocated, truncated at the end) ---
    u_history     = zeros(n, 100);
    v_history     = nan(n, 100);
    theta_history = nan(100, 1);
    u_history(:, 1) = u;
    % v_history(:, 1) stays NaN (no v at init)
    % theta_history(1) stays NaN
    % hal(1) is already 0 (zeros buffer)

    % --- 外層 Newton 迴圈 (Multi.m line 22-52) ---
    while min(res) > tol*(na*norm(u)+nb) && nit < 100

        nit = nit + 1;

        %%% solve linear system
        M   = sp_Jaco_Ax(AA,u,m)/(m-1);
        v   = M\b;

        %%% Newton step θ = 1
        theta = 1;
        tol_theta = 1e-14;
        u_old = u; v_old = v;
        u     = (1-theta/(m-1))*u + theta * v/(m-1) ;
        temp  = tpv(AA,u,m);
        res(nit) = norm( temp - b );
        hit = 0;

        %%% one-third halving procedure
        while res(nit) - res(nit-1) > 0 || min(temp) < 0
            theta    = theta/3;
            u        = (1-theta/(m-1))*u_old + theta * v_old/(m-1) ;
            temp     = tpv(AA,u,m);
            res_new  = norm( temp - b );
            res(nit) = res_new;
            hit      = hit+1;
            if theta < tol_theta
                fprintf('Can''t find a suitible step length such that inner residual decrease!');
                break;
            end
        end
        hal(nit) = hit;

        %%% record per-iter history
        u_history(:, nit)  = u;
        v_history(:, nit)  = v;
        theta_history(nit) = theta;
    end

    % --- truncate to the number of slots actually used ---
    u_history     = u_history(:, 1:nit);
    v_history     = v_history(:, 1:nit);
    theta_history = theta_history(1:nit);
    res_history   = res(1:nit);
    hal_history   = hal(1:nit);

end


%%% === local functions (verbatim from Multi.m lines 60-87) ===

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
