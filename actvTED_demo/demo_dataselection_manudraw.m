%=====================================================================
% Generate the toy data 
%=====================================================================
% clear all;      
% close all; 

X = drawdata;

N = size(X, 1);

f = figure(1);  
clf;
set(f, 'Position',  [410 214 740 620]); 
set(gca, 'FontSize', 16);
a = plot(X(:, 1), X(:, 2), 'b.' , 'MarkerSize', 10); 
hold on;   
axis([-1, 1, -1, 1]);
title('Toy Data (Press any key to continue)');
pause;

%=====================================================================
% compute the kernel matrix
%=====================================================================
sigma = 0.3; 
K = rbfkernel(X, X, sigma); 
    
%*************************************************************************
%  demo for one by one selection
%*************************************************************************
sample_size = min(10, N);
selected_data = X(transdesign_sq(K, sample_size), :);

%*************************************************************************
%  make a 2-D grid for visualization
%*************************************************************************
x_gridsize = 30;
y_gridsize = 30;
range = [-1, -1, 1, 1];
[testX, x_grid, y_grid] = grid_2d(range, x_gridsize, y_gridsize);

for i = 1:size(selected_data, 1);
    K_train = rbfkernel(selected_data(1:i,:), selected_data(1:i,:), sigma);
    K_gridtrain = rbfkernel(testX, selected_data(1:i,:), sigma);
    K_grid = rbfkernel(testX, testX, sigma) - K_gridtrain * inv(K_train) * K_gridtrain';
    clf;
    set(gca, 'FontSize', 16);
    funcvisual_2dgrid(diag(K_grid), x_grid, y_grid);
    colorbar;
    hold on;
    b = plot(selected_data(1:i,1), selected_data(1:i,2), 'r<', 'MarkerSize', 8);
    set(b, 'MarkerFace', 'r','LineWidth',1.5);
    a = plot(X(:, 1), X(:, 2), 'g.' , 'MarkerSize', 12); 
    axis([-1, 1, -1, 1]);
    hold off;
    title('Predictive Variance (Press any key to continue)');
    pause;
end
title('End!');
hold off;

