function [data, x_grid, y_grid] = grid_2d(range, xn, yn)

% Usge: X = grid_2d(range, xn, yn)
% Useful in visualizing 2-D functions
%
% range = [x_left, y_bottom, x_right, y_top] specifies the range of 2D grid
% xn and yn specify the density of grid
%
% example:   X = grid_2d(-1, -1, 1, 1, 15, 15) generates a 15x15 grid


x_left = range(1);
y_bottom = range(2);
x_right = range(3);
y_top = range(4);

x_grid = x_left:((x_right-x_left)/(xn - 1)):x_right;
y_grid = y_bottom:((y_top-y_bottom)/(yn - 1)):y_top;
[x, y] = meshgrid(x_grid, y_grid);
data = [x(:), y(:)];