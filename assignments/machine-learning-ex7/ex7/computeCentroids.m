function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

% for-loop over the centroids
for k = 1:K
    % number of x examples that were atributed to centroid k
    number_of_x_to_k = 0;
    % initialises to zeros with the size of the number of features
    sum = zeros(n, 1);
    for i = 1:m
        % only does the calculation on the x that is defined to that
        % centroid K
        if ( idx(i) == k )
            % gets the values of X in index i and sums that to the rest
            sum = sum + X(i, :)';
            % increments number of examples atributed to centroid k
            number_of_x_to_k = number_of_x_to_k + 1;
        end
    end
    % saves the new values of the centroid 
    centroids(k, :) = (1/number_of_x_to_k)*sum;
end






% =============================================================


end

