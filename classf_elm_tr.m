function [model] = classf_elm_tr(X,Y,param)
%Train a basic Extreme Learning Machine classfier.
%	X : each row is a sample.
%	Y : a column vector, class labels for X starting from 1.
%	PARAM: struct of parameters. The beginning part of this code (before 
% defParam) explains each parameter, and also sets the default parameters. 
% You can change parameter p to x by setting PARAM.p = x. For parameters
% that are not set, default values will be used.
% Return:
%	MODEL: a struct containing coefficients.
% 
%	Derived from Huang's Extreme Learning Machine code.
%%%%    Authors:    MR QIN-YU ZHU AND DR GUANG-BIN HUANG
%%%%    NANYANG TECHNOLOGICAL UNIVERSITY, SINGAPORE
%%%%    EMAIL:      EGBHUANG@NTU.EDU.SG; GBHUANG@IEEE.ORG
%%%%    WEBSITE:    http://www.ntu.edu.sg/eee/icis/cv/egbhuang.htm
%%%%    DATE:       APRIL 2004

X = X';
nSmp = size(X,2);
nFt = size(X,1);

hidNum = ceil(nFt*1)*3; % Number of hidden neurons assigned to the ELM
% actFunName - Type of activation function:
%   'sig' for Sigmoidal function
%   'sin' for Sine function
%   'hardlim' for Hardlim function
%   'tribas' for Triangular basis function
%   'radbas' for Radial basis function (for additive type of SLFNs instead of RBF type of SLFNs)
actFunName = 'lin';
C = .02; % regulation parameter, OutputWeight = (eye(size(H,1))/C+H * H') \ H * target';

defParam

target = ind2vec(Y');
target = target*2-1;

%%%%%%%%%%% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
InputWeight = rand(hidNum,nFt)*2-1;
BiasofHiddenNeurons = rand(hidNum,1);
tempH = InputWeight*X;
tempH = tempH+repmat(BiasofHiddenNeurons,1,nSmp);

%%%%%%%%%%% Calculate hidden neuron output matrix H
switch lower(actFunName)
	case {'sig','sigmoid'}
		actFun = @(x)1 ./ (1 + exp(-x));
	case {'sin','sine'}
		actFun = @(x)sin(x);
	case {'hardlim'}
		actFun = @(x)double(hardlim(x));
	case {'tribas'}
		actFun = @(x)tribas(x);
	case {'rbf','radbas'}
		actFun = @(x)radbas(x);
	case {'lin'}
		actFun = @(x)x;
end
H = actFun(tempH);

%%%%%%%%%%% Calculate output weights OutputWeight (beta_i)
% 	OutputWeight = pinv(H') * target';% without regularization factor //refer to 2006 Neurocomputing paper
OutputWeight = (eye(size(H,1))/C+H * H') \ H * target';   % faster method 1 //refer to 2012 IEEE TSMC-B paper
%If you use faster methods or kernel method, PLEASE CITE in your paper properly:
%Guang-Bin Huang, Hongming Zhou, Xiaojian Ding, and Rui Zhang, "Extreme Learning Machine for Regression and Multi-Class Classification," submitted to IEEE Transactions on Pattern Analysis and Machine Intelligence, October 2010.
trainOutput = H'*OutputWeight;

model.InputWeight = InputWeight;
model.actFun = actFun;
model.BiasofHiddenNeurons = BiasofHiddenNeurons;
model.OutputWeight = OutputWeight;

end