function funcvisual_2dgrid(Y, x_grid, y_grid, level)

% Usage: funcvisual_2dgrid(Y, x_grid, y_grid)
% Visualize a function on an two dimensional grid with smoothing. 
% Gray level as well as contour lines are visualized. This
% function is closely cooperated by grid_2d.m
%
% Y: a 1xN^2 column vector, function values of data points on the 2-D grid
% x_grid: 1xN vector, grid on the X axis
% y_grid: 1xN vector, grid on the Y axis




error(nargchk(3, 4, nargin));
if nargin < 4
    level = [];
end 
if isempty(level)
    level = 9;
end



axis([min(x_grid) max(x_grid) min(y_grid) max(y_grid)]);
imag = reshape(Y, length(y_grid), length(x_grid));
%axis('xy')
%colormap(gray);
colormap(bone);
pcolor(x_grid, y_grid, imag);
hold on;
shading interp
contour(x_grid, y_grid, imag, level, 'b');
hold off;

  