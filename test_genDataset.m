function [ X,Y ] = test_genDataset( name )
%Generate datasets for test.m

switch name
	case 'classf ionosphere'
		load ionosphere
		Y = (cell2mat(Y)=='b')+1; % text label to number
	case 'binary-class two gauss'
		d = 100; n = 50; s = 10;
		X = [mvnrnd(randn(1,d),eye(d)*s,n);mvnrnd(randn(1,d),eye(d)*s,n);
			mvnrnd(randn(1,d),eye(d)*s,n);mvnrnd(randn(1,d),eye(d)*s,n)];
		Y = [ones(n*2,1);ones(n*2,1)*2];
	case 'multi-class two gauss'
		d = 100; n = 50; s = 10;
		X = [mvnrnd(randn(1,d),eye(d)*s,n);mvnrnd(randn(1,d),eye(d)*s,n);
			mvnrnd(randn(1,d),eye(d)*s,n);mvnrnd(randn(1,d),eye(d)*s,n);
			mvnrnd(randn(1,d),eye(d)*s,n);mvnrnd(randn(1,d),eye(d)*s,n)];
		Y = [ones(n*2,1);ones(n*2,1)*2;ones(n*2,1)*3];
	case 'regression synthetic'
		d = 10; n = 100; s = .1;
		X = randn(n,d); % useful features 1:d
		Y = X*rand(d,1)+randn(n,1)*s;
		X = [X,randn(n,2)]; % noise features
	case 'regression spectra'
		load spectra
		Y = octane;
		X = NIR;
	case 'ftSel synthetic'
		d = 10; n = 100; s = 1;
		X = randn(n,d); % useful features 1:d
		Y = double((X*rand(d,1)+rand(n,1)*s)>0)+1;
		X = [X,randn(n,d)]; % noise features d+1:d*2
end

fprintf(['using dataset "',name,'" ...\n'])

end

