% Verbatim from matlab_ref/hni/HONI.m line 129-139.
% Do not edit — this is kept in sync with the canonical source.

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
