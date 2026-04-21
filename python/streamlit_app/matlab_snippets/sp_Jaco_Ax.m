% Verbatim from matlab_ref/hni/Multi.m line 79-87.
% Do not edit — this is kept in sync with the canonical source.

function J = sp_Jaco_Ax( AA, x, m )
% here return a matrix F'(x) where F(x)=Ax^(m-1)
    I = speye(length(x));
    J = 0;
    p = m-1;
    for i = 1:p
        J = J + AA*kron( tenpow(x,i-1) , kron( I , tenpow(x,p-i) ) );
    end
end
