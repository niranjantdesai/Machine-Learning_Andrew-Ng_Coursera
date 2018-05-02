function [C, sigma] = dataset3Params(X, y, Xval, yval, x1, x2)
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

CVal = [0.01 0.03 0.1 0.3 1 3 10 30];
sigmaVal = [0.01 0.03 0.1 0.3 1 3 10 30];
error = zeros(size(CVal,2),size(sigmaVal,2));
for i=1:size(CVal,2)
   for j=1:size(sigmaVal,2)
       model= svmTrain(X, y, CVal(i), @(x1, x2) gaussianKernel(x1, x2, sigmaVal(j)));
       predictions = svmPredict(model,Xval);
       error(i,j) = mean(double(predictions ~= yval));
   end
end

[~,I] = min(error(:));
[I_row, I_col] = ind2sub(size(error),I);
C = CVal(I_row);
sigma = sigmaVal(I_col);
% =========================================================================

end
