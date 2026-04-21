% Verbatim from matlab_ref/hni/HONI.m line 152-177.
% Do not edit — this is kept in sync with the canonical source.

function B = ten2mat( A , k)
% Construct k-type matrix form of a tensor A, k must less than the order of
% tensor A.
    n = size(A,1);
    m = length(size(A));

    [temp1,temp2] = idx_create( n, k );
    express = ['reshape(','A(',temp1,'i',temp2,')',',',int2str(1),',',int2str(n^(m-1)),');'];
    
    B = zeros(n,n^(m-1));
    for i = 1:n
        B(i,:) = eval( express );
    end
end

function [temp1,temp2] = idx_create( n, type )
    temp1 = [];
    temp2 = [];
    for i = 1:n
        if i < type
            temp1 = [temp1,':,'];
        elseif i > type
            temp2 = [temp2,',:'];
        end
    end
end
