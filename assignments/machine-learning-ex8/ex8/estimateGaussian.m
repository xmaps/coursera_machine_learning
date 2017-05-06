function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%

% S = mean(X) is the mean value of the elements in X if X is a vector. 
%    For matrices, S is a row vector containing the mean value of each 
%    column.
mu = mean(X)';

% For vectors, Y = var(X) returns the variance of the values in X.  For
% matrices, Y is a row vector containing the variance of each column of
% X.  For N-D arrays, var operates along the first non-singleton
% dimension of X.
sigma2 = var(X,1)';


% =============================================================


end
