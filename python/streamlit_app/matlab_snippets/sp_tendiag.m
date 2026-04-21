% Verbatim from matlab_ref/hni/HONI.m line 142-149.
% Do not edit — this is kept in sync with the canonical source.

function D = sp_tendiag( d, m )
% Construct m-order, n-dimension diagonal tensor with diagonal entrices d .
    n    = length(d);
    D    = sparse(n^m,1);
    S    = linspace(1,n^m,n); 
    D(S) = d;
    D    = reshape(D,n,n^(m-1));
end    
