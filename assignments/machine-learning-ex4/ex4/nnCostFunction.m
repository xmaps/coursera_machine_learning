function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Part 1 -----------------------------------------------------------------

% Part 1.1 - initizalize the Y matrix

% instead of a y vector of 0-9 values we need a 10-dimensional vector 
% so the Y will be a matrix where each row is the vector corresponding to 
% correct class

% each row corresponds to the class where for example 
% y5 = 1, and the other elements equal to 0
classes_identity_matrix = eye(num_labels);
% initialize Y where we have a m training examples and K classes of zeros
% and ones
Y = zeros(m, num_labels);
% in each training example gets the value y and maps to a vector of zeros
% and ones and saves to the new Y matrix
for i=1:m
  Y(i, :)= classes_identity_matrix(y(i), :);
end

% Part 1.2 - feedfoward to get the hypostesis to calculate the Cost

% initialise the input layer and add ones to the first column to be the
% bias
a1 = [ones(m, 1) X];

% hidden layer (2nd layer)
z2 = a1*Theta1';
a2_intermediate = sigmoid(z2);
% add the bias unit to the matrix. the size changes to the number of units 
a2 = [ones(size(z2, 1), 1) a2_intermediate];

% output layer
z3 = a2*Theta2';
a3 = sigmoid(z3);

% Part 1.3 - Calculate Cost Function

hypotesis = a3;

% cost function
first_part = (-Y).*log(hypotesis);
second_part = (1-Y).*log(1-hypotesis);
J = (1/m) * (sum(sum(first_part - second_part)));

% Part 1.4 - Add regularization to the Cost Function

% You should not be regularizing the terms that correspond to the bias. 
% For the matrices Theta1 and Theta2, this corresponds to the first column 
% of each matrix.
theta1_without_first_column = Theta1(:, 2:end);
theta2_without_first_column = Theta2(:, 2:end);

first_theta1_reg = sum(sum(theta1_without_first_column.^2));
first_theta2_reg = sum(sum(theta2_without_first_column.^2));

regularization = (lambda/(2*m))*(first_theta1_reg + first_theta2_reg);

J = J + regularization;


% Part 2 -----------------------------------------------------------------

% vectorized implementation removing the need to a for loop

% Part 2.1 - output layer

sigma_3 = hypotesis-Y;

% Part 2.2 - hidden layer

% add the bias unit to the matrix. the size changes to the number of units
z2_with_bias = [ones(size(z2, 1), 1) z2];
% sigma calculation
sigma_2 = (sigma_3*Theta2).*sigmoidGradient(z2_with_bias);

% Part 2.3 - Accumulate gradients

% you should skip or remove delta_2
sigma_2 = sigma_2(:, 2:end);

delta_1 = (sigma_2'*a1);
delta_2 = (sigma_3'*a2);

% Part 2.4 - unregularized gradient

Theta1_grad = (1/m)*delta_1;
Theta2_grad = (1/m)*delta_2;


% Part 3 -----------------------------------------------------------------

% Part 3.1 - you should not be regularizing the first column of thetas which
% is used for the bias term.
modified_theta1 = [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
modified_theta2 = [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];

% Part 3.2 - calculate regularided part
p1 = (lambda/m)*modified_theta1;
p2 = (lambda/m)*modified_theta2;

% Part 3.3 - add regularized part to grads
Theta1_grad = ((1/m)*delta_1) + p1;
Theta2_grad = ((1/m)*delta_2) + p2;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
