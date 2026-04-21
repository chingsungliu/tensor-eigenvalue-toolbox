clear 
clc

%% Setting parameters
m            =  3;         % tensor order
n            = 20;         % tensor dimension

%% Generate basic tensors or a n-by-n^(m-1) stretching matrix of tensors

Q = rand(n*ones(1,m));  %% mth order n-dimension tensor 
%Q = rand(n,n^(m-1));   %% n-by-n^(m-1) matrix


%% Test eigensolvers
%%parameters setting
x_0 = abs(ones(n,1));
plot_res  = 0;
tolerance = 1e-11;    
option.initial_vector = x_0;            % initial vector for iterations
option.maxit          = 2000;           % stopping criterion respect to iteration number

%%% for using HONI
option.linear_solver  = 'exact';        % exact solver for the multilinear system 
[  EV1, EW1, res1, out1, innit1, halv1 ] = HONI( Q, plot_res, tolerance, option);

%%% for using Inexact HONI
option.linear_solver  = 'inexact';     % inexact solver for the multilinear system 
[  EV2, EW2, res2, out2, innit2, halv2 ] = HONI( Q, plot_res, tolerance, option);

  






    
