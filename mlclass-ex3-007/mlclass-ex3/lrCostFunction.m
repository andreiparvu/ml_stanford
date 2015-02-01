function [J, grad] = lrCostFunction(theta, X, y, lambda)
  %LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
  %regularization
  %   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
  %   theta as the parameter for regularized logistic regression and the
  %   gradient of the cost w.r.t. to the parameters. 

  % Initialize some useful values
  m = length(y); % number of training examples

  n = size(X, 2);
  % You need to return the following variables correctly 
  J = 0;
  grad = zeros(size(theta));

  sm = -y .* log(sigmoid(X * theta)) - (1 - y) .* log(1 - sigmoid(X * theta));

  J = 1 / m * sum(sm) + lambda / (2 * m) * sum(theta(2 : size(theta)) .^ 2);

  grad(1) = 1 / m * sum((sigmoid(X * theta) - y) .* X(:, 1));

  grad(2 : n) = 1 / m * sum((sigmoid(X * theta) - y) .* X(:, 2 : n))' + ...
                lambda / m * theta(2 : n);
end
