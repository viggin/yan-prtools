function [model] = classf_tree_tr(X,Y,param)
%Wrapper of ClassificationTree class in matlab.
% 
% Can search the doc of ClassificationTree for further configurations.
%	X : each row is a sample.
%	Y : a column vector, class labels for X starting from 1.
%	PARAM: struct of parameters. Please see the doc of ClassificationTree
% and add your parameters in the code.
% Return:
%	MODEL: a struct containing coefficients.
% 
%	Ke YAN, 2016, Tsinghua Univ. http://yanke23.com, xjed09@gmail.com

defParam

model = ClassificationTree.fit(X,Y);

end
