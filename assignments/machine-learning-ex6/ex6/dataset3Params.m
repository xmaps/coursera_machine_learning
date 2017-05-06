function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


% inf returns the IEEE arithmetic representation for positive infinity. So
% the biggest possible value for error
minimum_error = inf;
% suggested values
values = [0.01 0.03 0.1 0.3 1 3 10 30];

for temp_C = values
  for temp_sigma = values
    % train the SVM model
    model = svmTrain(X, y, temp_C, @(x1, x2) gaussianKernel(x1, x2, temp_sigma));
    % svmPredict to predict the labels on the cross validation set.    
    predictions = svmPredict(model, Xval);
    % compute the prediction error
    prediction_error = mean(double(predictions ~= yval));
    % change values if the the error is lower than the previous minimum
    if( prediction_error <= minimum_error )
      C = temp_C;
      sigma = temp_sigma;
      minimum_error = prediction_error;
    end
  end
end

% =========================================================================

end
