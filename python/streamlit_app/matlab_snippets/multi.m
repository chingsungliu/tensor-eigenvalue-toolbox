function [u,nit,hal] = Multi(AA,b,m,tol)

%   Multi:  Find the solution u of a multilinear system AAu^(m-1)=b.
%   Input:
%   AA  =  a n-by-n^(m-1) stretching matrix of M-tensor for the multilinear system.
%   b   =  the right hand side of the multilinear system
%   tol =  the stopping criterion of Multi process.
%   
%   Output:
%   u   = return the positive solution of a multilinear system.
%   nit = the total number of iterations to achieve convergence.
%   hit = the total number of one-third procedure.

u      = b/norm(b);
b      = b.^(m-1);
na     = sqrt(norm(AA,'inf')*norm(AA,1));
nb     = norm(b);
temp   = tpv(AA,u,m);
res    = (na+nb)*ones(100,1);
res(1) = norm( temp - b  );
hal    = zeros(100,1);nit = 1;
while   min(res) > tol*(na*norm(u)+nb) && nit < 100 

    nit = nit + 1;
    
    %%% solve linear system
    M   = sp_Jaco_Ax(AA,u,m )/(m-1);
    v   = M\b;
     
    %%% update the solution of the multilinear system
    theta = 1;
    tol_theta=1e-14;
    u_old = u; v_old=v;
    u     = (1-theta/(m-1))*u + theta * v/(m-1) ;
    temp  =  tpv(AA,u,m);
    res(nit) = norm( temp - b  );
    hit = 0;
    %%%  one-third procedure
        while res(nit) - res(nit-1) > 0 || min(temp) < 0
          theta    = theta/3;
          u        = (1-theta/(m-1))*u_old + theta * v_old/(m-1) ;
          temp     = tpv(AA,u,m);
          res_new  = norm( temp - b  );
          res(nit) = res_new;
          hit      = hit+1;
          if theta < tol_theta %|| min(u)<0
                fprintf('Can''t find a suitible step length such that inner residual decrease!')
                break;
          end 
        end
        hal(nit)=hit;
end

end




    %% tensor product with vector
function y = tpv( AA ,x, m )
% Compute the m-tensor product with vector : Ax^(m-1)
    x_m = tenpow(x,m-1);
    y   = AA*x_m;
end

function x_p = tenpow( x,p )
% Compute x^(p) = x@x@...@x  (p-times and '@' denote kronecker product)
    if p == 0
        x_p = 1;
    else
        x_p = x;
        for i = 1:p-1
            x_p = kron(x,x_p);
        end
    end
end

%% sparse Jaconi form of x^p
function J = sp_Jaco_Ax( AA, x, m )
% here return a matrix F'(x) where F(x)=Ax^(m-1)
    I = speye(length(x));
    J = 0;
    p = m-1;
    for i = 1:p
        J = J + AA*kron( tenpow(x,i-1) , kron( I , tenpow(x,p-i) ) );
    end
end

