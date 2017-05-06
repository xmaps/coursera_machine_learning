function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

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

% returns the maximum probability and the index of it
[p_max, i_max] = max(a3, [], 2);

% indice of the max corresponds to the y (e.g. 0-9) with the maximum
% probability
p = i_max;

% =========================================================================


end
