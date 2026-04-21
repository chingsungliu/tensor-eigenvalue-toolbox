% Verbatim from matlab_ref/hni/HONI.m line 123-127.
% Do not edit — this is kept in sync with the canonical source.

function y = tpv( AA ,x, m )
% Compute the m-tensor product with vector : Ax^(m-1)
    x_m = tenpow(x,m-1);
    y   = AA*x_m;
end
