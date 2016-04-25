%=====================================================================
% Generate the toy data and visualize them
%=====================================================================
clear all; 
close all; 

X = fourgaussian;
N = size(X, 1);

f = figure(1);  
clf;
set(f, 'Position',  [410 214 690 520]); 
set(gca, 'FontSize', 16);
set(gcf, 'DefaultAxesLineWidth',1);
a = plot(X(:, 1), X(:, 2), 'k.' , 'MarkerSize', 12); 
hold on;   
axis([-7, 11, -7, 12]);
title('Toy Data (Press any key to continue)');
pause;

%=====================================================================
% compute the kernel matrix
%=====================================================================
sigma = 1.8; 
K = rbfkernel(X, X, sigma);   %RBF kernel matrix
    
%*************************************************************************
%  Sequential transductive experimental desgin
%*************************************************************************
sample_size = min(4, N);       % selecting 4 data samples
selected_data = X(transdesign_sq(K, sample_size), :);

%*************************************************************************
%  visualize the predictive variance with data selected
%*************************************************************************
x_gridsize = 30;
y_gridsize = 30;
range = [-7, -7, 11, 12];
[testX, x_grid, y_grid] = grid_2d(range, x_gridsize, y_gridsize);
for i = 1:size(selected_data, 1);
    K_train = rbfkernel(selected_data(1:i,:), selected_data(1:i,:), sigma);
    K_gridtrain = rbfkernel(testX, selected_data(1:i,:), sigma);
    K_grid = rbfkernel(testX, testX, sigma) - K_gridtrain * inv(K_train) * K_gridtrain';
    figure(1)
    clf;
    set(gca, 'FontSize', 16);
    set(gcf, 'DefaultAxesLineWidth',1);
    funcvisual_2dgrid(diag(K_grid), x_grid, y_grid);
    colorbar;
    hold on;
    a = plot(X(:, 1), X(:, 2), 'k.' , 'MarkerSize', 12); 
    b = plot(selected_data(1:i,1), selected_data(1:i,2), 'r<', 'MarkerSize', 12);
    set(b, 'MarkerFace', 'r','LineWidth',1.5);
    axis([-7, 11, -7, 12]);
    hold off;
    title('Predictive Variance (Press any key to continue)');
    pause;
end
hold off;
title('Predictive Variance (End!)');

