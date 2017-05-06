function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

hypotesis = sigmoid(X*theta);

% put the first theta a zero and then get all row starting at index 2
all_thetas_expect_zero = [0 ; theta(2:size(theta), :)];

% regularization part (transpose thetas to make it a n*1 matrix instead of a vector 1*n)
regularization = (lambda/(2*m))*(all_thetas_expect_zero'*all_thetas_expect_zero);

% cost function
first_part = (-y)'*log(hypotesis);
second_part = (1-y)'*log(1-hypotesis);
J = (1/m) * (first_part - second_part) + regularization;

% calculate grads
grad = ((1/m) * (X'*(hypotesis-y))) + ((lambda/m)*all_thetas_expect_zero);

% =============================================================

end
