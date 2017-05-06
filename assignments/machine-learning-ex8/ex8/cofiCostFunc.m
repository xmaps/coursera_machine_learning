function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

squared_distance = (X*Theta'-Y).^2;
% multipling the distances by only the movies evaluated by the user
J = (1/2)*sum(squared_distance(R==1));

% cost function regularized part
regularized_term_theta = (lambda/2)*sum(sum(Theta.^2));
regularized_term_x = (lambda/2)*sum(sum(X.^2));
J = J + regularized_term_theta + regularized_term_x;


for i=1:num_movies
    % list of all the users that have rated movie i.
    idx = find(R(i, :)==1);
    % user features of movie i with rating
    theta_temp = Theta(idx, :);
    % ratings of movie i
    y_temp = Y(i, idx);
    % calc the derivatives
    X_grad(i, :) = (X(i, :)*theta_temp' - y_temp) * theta_temp;
    % regularized X
    X_grad(i, :) = X_grad(i, :) + (lambda*X(i, :));
end

for j=1:num_users
    % list of all the movies rated by user j.
    idx = find(R(:, j)==1);
    % movie features of user j with rating
    x_temp = X(idx, :);
    % ratings by user j
    y_temp = Y(idx, j);
    % calc the derivatives
    Theta_grad(j, :) = (x_temp*Theta(j, :)' - y_temp)' * x_temp;
    % regularized Theta
    Theta_grad(j, :) = Theta_grad(j, :) + (lambda*Theta(j, :));
end










% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
