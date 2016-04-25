function [ft] = ftProc_mat2ftvec(data)
%Transform sample matrices to a feature matrix.
%	In pattern recognition, sometimes each sample is a matrix. We need to 
% first reshape each sample to a row vector.
%	DATA:	a cell array of matrices; or a matrix (only 1 sample)
% Return:
%	FT:		if DATA is a matrix, FT=DATA(:)'; if DATA is a matrix, FT(i,:)=DATA{i}(:)'
% 
%	Ke YAN, 2016, Tsinghua Univ. http://yanke23.com, xjed09@gmail.com

if iscell(data) % if each sample is a matrix in a cell
	[nRow, nCol] = size(data{1});
	nSmp = length(data);
	ft = zeros(nSmp,nRow*nCol);
	for iSmp = 1:nSmp % transform the matrix to a vector
		ft(iSmp,:) = data{iSmp}(:)';
	end
else % only 1 sample
	ft = data(:)';
end

end