function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

hypotesis = X * theta;

% put the first theta a zero and then get all row starting at index 2
all_thetas_expect_zero = [0 ; theta(2:size(theta), :)];

% regularization part (transpose thetas to make it a n*1 matrix instead of a vector 1*n)
regularization = (lambda/(2*m))*(all_thetas_expect_zero'*all_thetas_expect_zero);

% cost function
sqrErrors = (hypotesis - y).^2;
J = 1/(2*m) * sum(sqrErrors) + regularization;

% calculate grads
grad = ((1/m) * (X'*(hypotesis-y))) + ((lambda/m)*all_thetas_expect_zero);


% =========================================================================

grad = grad(:);

end
