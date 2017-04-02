function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% multiply the X matrix (97x2) with the theta vector (2x1) to get new a 
% vector containing the hypothesis values for each row in X
hypothesis = X * theta;

% subtract the y vector from the hypothesis vector, resulting in a vector
% containing the difference in values for each data row
differenceFromY = hypothesis - y;
squaredError = differenceFromY.^2;

J = sum(squaredError)/(2 * m); % return the sum of all errors as the cost function

% =========================================================================

end
