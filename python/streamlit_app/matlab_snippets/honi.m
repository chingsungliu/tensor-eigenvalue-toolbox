 function varargout = HONI( varargin )
 
 
%   HONI  Find the largest eigenvalue and corresponding eigenvectors of a tensor.
%   mu = HONI(A) returns the largest eigenvalue of A.
%   A can be either a m-order, n-dimensional square tensor or a n-by-n^(m-1) 
%   stretching matrix of this tensor.
%
%   [v,mu] = HONI(A) returns the largest eigenvalue and corresponding
%   eigenvector of A.
%   [v,mu,res,nit] = HONI(A) also returns the corresponding residual and the total number of outer terations.
%   [v,mu,res,nit,innit,hal] = HONI(A) also returns the total number of inner terations 
%    and the total number of halving precedure.
%
%
%   HONI(A,plot_res) can see the residual in each iteration if set
%   plot_res=1, the defalt set is zero.
%
%   HONI(A,plot_res,tolerence) tolerance is the stopping criterion of HONI
%   process.
%
%   HONI(A,plot_res,tolerence,option) option assigned be a structure cell which 
%   contains fields 'linear_solver' and 'initial_vector'.
%   'linear_solver'can be use to choose the exact solver or inexact solver
%   for solving the linear system for the multilinear systems, as follows
%
%   option.linear_solver = 'solver name'
%   The 'solver name' can be choosen between 'exact'(exact solver, default) and 'inexact'(inexact solver).
%
%   'initial_vector' can be use to assign the initial vector in the
%   begining of iteration. the default setting is a random vector.

   [ AA, m, n, plot_res, tolerance, option ] = input_check( nargin, nargout, varargin );
    
    x         = option.initial_vector;
    x         = x/norm(x);
    temp      = tpv(AA,x,m)./(x.^(m-1));
    lambda_U  = max(temp);
    II= sp_tendiag( ones(n,1), m );
    maxit     = option.maxit;
    res       = ones(maxit,1);
    nit=1;% total number of outer iterations
    res(1) = abs( max(temp)-min(temp) )/lambda_U;
    hal=0;innit=0;
    
    while  min(res) > tolerance && nit < maxit
    nit      = nit + 1;
    
    if strcmp(option.linear_solver,'exact')
         %%% solve the multilinear system by using Newton's method.
         %%% the following four step can be replaced by any multilinear
         %%% solver
         inner_tol = 1e-10;
         [y,chit,hal_inn]  = Multi(lambda_U*II-AA,x,m,inner_tol);
         hal = hal + sum(hal_inn);    
         innit = innit +chit;
         %%% the update of the approximate eigenpair(s)          
         temp = x./y;
         lambda_U = lambda_U - min(temp)^(m-1); 
         res(nit,1) = abs( max(temp)^(m-1)-min(temp)^(m-1) )/lambda_U;
         x = y/norm(y);
          
    elseif strcmp(option.linear_solver,'inexact')
        %%% solve the multilinear system by using Newton' method.
         %%% the following four step can be replaced by any multilinear
         %%% solver
        inner_tol = max(1e-10,min(res)*min(x)^(m-1)/nit);
        [y,chit,hal_inn]  = Multi(lambda_U*II-AA,x,m,inner_tol);
        hal = hal + sum(hal_inn);      
        innit = innit +chit;
        %%% the update of the approximate eigenpair(s) 
        x = y/norm(y);
        temp         = tpv(AA,x,m)./(x.^(m-1));
        lambda_U = max( temp );
        res(nit,1) = abs( max(temp)-min(temp) )/lambda_U;
    end
 
   
    end
    
     lambda = lambda_U;
    if plot_res == 1
       % semilogy(1:nit,res(1:nit),'kx-');  
        loglog(1:nit,res(1:nit),'kx-'); 
        hold on
    end
    switch nargout
        case 0
            varargout{1} = lambda;
        case 1
            varargout{1} = lambda;
        case 2
            varargout{1} = x;
            varargout{2} = lambda;
        case 3
            varargout{1} = x;
            varargout{2} = lambda;
            varargout{3} = res(1:nit);
        case 4
            varargout{1} = x;
            varargout{2} = lambda;
            varargout{3} = res(1:nit);
            varargout{4} = nit;
        case 5
            varargout{1} = x;
            varargout{2} = lambda;
            varargout{3} = res(1:nit);
            varargout{4} = nit;
            varargout{5} = innit;
        case 6
            varargout{1} = x;
            varargout{2} = lambda;
            varargout{3} = res(1:nit);
            varargout{4} = nit;
            varargout{5} = innit;
            varargout{6} = hal;
    end
    
 end
   


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


function D = sp_tendiag( d, m )
% Construct m-order, n-dimension diagonal tensor with diagonal entrices d .
    n    = length(d);
    D    = sparse(n^m,1);
    S    = linspace(1,n^m,n); 
    D(S) = d;
    D    = reshape(D,n,n^(m-1));
end    
    

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



%% input check
function [ A,m,n, plot_res, tolerence, option ] = input_check(Nargin,Nargout,Varargin)
option_default       = struct('linear_solver','exact','initial_vector',[],'maxit',100);
    switch Nargin
        case {0} 
            error ('Not enough input!');
        case 1
            A            = Varargin{1};
            plot_res     = 0;
            tolerence    = 1e-12;
            option       = option_default;
        case 2
            A            = Varargin{1};
            plot_res     = Varargin{2};
            tolerence    = 1e-12;
            option       = option_default;
        case 3
            A            = Varargin{1};
            plot_res     = Varargin{2};
            tolerence    = Varargin{3};
            option       = option_default;
        case 4
            A            = Varargin{1};
            plot_res     = Varargin{2};
            tolerence    = Varargin{3};
            option       = Varargin{4};
        otherwise
            error('Too many input argument')
    end
    
    if length( size(A) ) == 2 % if input A is a matrix
        n = size(A,1);
        m = round( log( size(A,1)*size(A,2) ) / log(n) );
    elseif length( size(A) ) >= 2 % if input A is a tensor
        n           = size(A,1);
        m           = length( size(A) );
        prodct_type = 1;
        A           = ten2mat( A , prodct_type);
    end
    
    if isempty(option.linear_solver)
        option.linear_solver = 'exact';
    end
    if isempty(option.initial_vector)
        option.initial_vector = rand(n,1);
    end
    
    if Nargout > 6
        error('Too many output argument.');
    end
end

