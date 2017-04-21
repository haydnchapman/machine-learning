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

hypothesis = X * theta;

differenceFromY = hypothesis - y;

parameters = (theta(2:end));

unregularizedCost = (1 / (2 * m)) * sum(differenceFromY .^ 2);

regularizationCost = (lambda / (2 * m)) * sum(parameters .^ 2);

J = unregularizedCost + regularizationCost;

% standard calculation of non regularized gradients
grad = (X' * differenceFromY) / m;

% create a vector containing the regularized values for each parameter, with
% 0 set for the first row as we don't regularize theta zero
regularizationVector = [0; lambda * parameters / m];

% add the standard gradient vector with the regularization vector
grad = grad + regularizationVector;

% =========================================================================
grad = grad(:);

end
