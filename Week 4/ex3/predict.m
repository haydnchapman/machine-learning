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

% [(m)] vector containing the +1 bias values for each input row
bias = ones(size(X,1),1);

% form an [(m) x (n+1)] matrix containing all inputs and their bias
a1 = [bias,X];

z2 = a1 * Theta1';
% add bias parameter to each row and apply sigmoid function to z2 matrix,
a2 = [bias,sigmoid(z2)];

z3 = a2 * Theta2';
% apply sigmoid function to z3 matrix
a3 = sigmoid(z3);

% retrieve index of max value within a3 matrix
[~,index] = max(a3,[],2);

% Which in turn returns an [m] dimension vector representing the label 
% number that each row most closely matches
p = index;

% =========================================================================


end
