clear; clc; close all;

% ---  LOAD DATA ---
 isfile('MFCC_Features.mat');
 load('MFCC_Features.mat');

 required_fields = {'blues', 'reggae', 'classical'};

% --- GRAPH 1: FEATURE VISUALIZATION (HEATMAP) ---
% We plot a snippet of the MFCCs (e.g., first 600 frames) for one representative song/genre
fprintf('Generating Figure 1: MFCC Heatmaps...\n');
figure('Color', 'w', 'Position', [100, 100, 1200, 400]);

frames_to_plot = 600; 
genres_plot = {'blues', 'reggae', 'classical'};
titles = {'Blues Texture', 'Reggae Texture', 'Classical Texture'};

for i = 1:3
    g = genres_plot{i};
    data = train_data.(g);
    plot_data = data(1:frames_to_plot, :)';
    subplot(1, 3, i);
    imagesc(plot_data); 
    axis xy; 
    title(titles{i}, 'FontSize', 12, 'FontWeight', 'bold'); 
    xlabel('Time Frame (Samples)'); 
    ylabel('MFCC Coefficient Index');
    colormap(parula); 
    clim([-4 4]); 
    colorbar;
end

sgtitle('MFCC Feature Comparison (First 600 Frames)', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, 'mfcc_comparison.png'); 

% --- GRAPH 2: SCATTER PLOT (CLUSTERING) ---
% We plot Coefficient 2 vs Coefficient 3 to show class separation
fprintf('Generating Figure 2: 2D Scatter Plot...\n');
figure('Color', 'w', 'Position', [150, 150, 800, 600]);
hold on; grid on; box on;

% Number of random points to sample per genre (to avoid over-plotting)
N_points = 2000; 

% Genre Configurations
% Note: In final.m, we removed the 1st col (Energy). 
% So col 1 is now C1, col 2 is C2, col 3 is C3.
% We plot col 2 (x-axis) vs col 3 (y-axis).
configs = struct();
configs(1).genre = 'blues';     configs(1).color = 'b'; configs(1).shape = 's'; configs(1).name = 'Blues';
configs(2).genre = 'classical'; configs(2).color = 'r'; configs(2).shape = 'o'; configs(2).name = 'Classical';
configs(3).genre = 'reggae';    configs(3).color = 'g'; configs(3).shape = '^'; configs(3).name = 'Reggae';

for i = 1:3
    g = configs(i).genre;
    data = train_data.(g);
    
    % Randomly sample points
    total_frames = size(data, 1);
    if total_frames > N_points
        idx = randperm(total_frames, N_points);
        subset = data(idx, :);
    else
        subset = data;
    end
    
    scatter(subset(:, 2), subset(:, 3), 20, ...
        configs(i).color, configs(i).shape, 'filled', ...
        'MarkerFaceAlpha', 0.6, 'DisplayName', configs(i).name);
end

legend('Location', 'northeast', 'FontSize', 10);
xlabel('2^{nd} MFCC Coefficient', 'FontSize', 11);
ylabel('3^{rd} MFCC Coefficient', 'FontSize', 11);
title('2D Projection of Feature Space (Clustering)', 'FontSize', 14);
xlim([-5 5]); 
ylim([-5 5]);
saveas(gcf, 'feature_scatter.png');
