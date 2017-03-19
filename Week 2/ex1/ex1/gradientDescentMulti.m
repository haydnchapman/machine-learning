function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    
    % create a hypothesis matrix containing all values of X multiplied 
    % by the current theta
    hypothesis = X * theta;
    
    % create a matrix containing the difference between the hypothesis and
    % y values
    differencesFromY = hypothesis - y;
    
    % create a new 1x(n) matrix from the product of the transposed differencesFromY 
    % matrix ((1)x(m)) and X ((m)x(n+1)) => (1)x(m) * (m)x(n+1) => (1)x(n+1)
    % matrix
    delta = (differencesFromY' * X);
    
    % calculate the next version of theta using the transposed delta
    % matrix (to make it an (n)x(1) matrix, matching the theta matrix size
    theta = theta - alpha / m * delta'; 
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
