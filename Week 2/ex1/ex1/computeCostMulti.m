function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.


% =========================================================================

% multiply the X matrix (97x2) with the theta vector (2x1) to get new a 
% vector containing the hypothesis values for each row in X
hypothesis = X * theta;

% subtract the y vector from the hypothesis vector, resulting in a vector
% containing the difference in values for each data row
differenceFromY = hypothesis - y;

%using the forumla in the ex1.pdf for multivariate cost functions
multivariateCostFunction = (differenceFromY' * differenceFromY) / (2 * m);

J = multivariateCostFunction; 

end
