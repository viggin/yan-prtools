function model = regress_ann_tr(X,Y,param)
%Train an artificial neural network regressor.
%	There are many network functions in Matlab. You can choose what's best
% 
%	X : each row is a sample.
%	Y : a column vector.
%	PARAM: struct of parameters. The beginning part of this code (before 
% defParam) explains each parameter, and also sets the default parameters. 
% You can change parameter p to x by setting PARAM.p = x. For parameters
% that are not set, default values will be used.
% Return:
%	MODEL: a struct containing coefficients.
% 
%	Ke YAN, 2016, Tsinghua Univ. http://yanke23.com, xjed09@gmail.com

elNum = 6;
hidFcn = 'tansig'; % choose 'purelin' if you want to avoid overfitting
epochs = 100;
goal = 1e-1;

defParam

% net = newff(X,Yvec,elNum,{hidFcn,'purelin'},'trainlm'); % old style
model = feedforwardnet(elNum);
model.layers{1}.transferFcn = hidFcn;
model.trainParam.showWindow = 0;
model.trainParam.epochs = epochs;
model.trainParam.goal = goal;
model = train(model,X',Y');

% another model
% net = newrbe(X,Y);
% view(net)

end