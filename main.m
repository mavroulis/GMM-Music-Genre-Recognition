clear; clc; close all;

%  CONFIGURATION
genres = {'blues', 'reggae', 'classical'};
num_gaussians = [8, 16]; % Experimenting with Order 8 and 16
train_folder = 'Data/Train/'; 
test_folder = 'Data/Test/';   

disp('--- Extracting Features ---');
train_data = struct(); 
test_data = struct();

%  Extract Training Data
for i = 1:length(genres)
    genre = genres{i};
    folder_path = fullfile(train_folder, genre);
    fprintf('Processing Train Genre: %s\n', genre);
    train_data.(genre) = extract_genre_features(folder_path, 'train');
end

% Extract Testing Data
for i = 1:length(genres)
    genre = genres{i};
    folder_path = fullfile(test_folder, genre);
    fprintf('Processing Test Genre: %s\n', genre);
    test_data.(genre) = extract_genre_features(folder_path, 'test');
end

fprintf('Saving features to MFCC_Features.mat ... ');
save('MFCC_Features.mat', 'train_data', 'test_data');
fprintf('Done.\n');
    
%  TRAINING & TESTING LOOP 
for M = num_gaussians
    fprintf('\n=========================================\n');
    fprintf(' Experiment: GMM Order M = %d \n', M);
    fprintf('=========================================\n');
    
    models = struct();
    
    %  Train GMMs for each genre
    for i = 1:length(genres)
        g = genres{i};
        fprintf('Training GMM for %s (EM Algorithm)...\n', g);
        [w, mu, sigma] = train_gmm(train_data.(g), M);
        models.(g).w = w;
        models.(g).mu = mu;
        models.(g).sigma = sigma;
    end
    
    model_filename = sprintf('GMM_Parameters_M%d.mat', M);
    save(model_filename, 'models');
    fprintf('>> Saved trained models to %s\n', model_filename);
   
    %  Classification (MAP/Likelihood)
    correct = 0;
    total = 0;
    
    disp('--- Classification Results ---');
    fprintf('%-12s | %-12s | %s\n', 'True Genre', 'Predicted', 'Scores (Blues / Reggae / Classical)');
    fprintf('-------------------------------------------------------------\n');
    
    for i = 1:length(genres)
        true_genre = genres{i};
        songs = test_data.(true_genre); 
        
        for s = 1:length(songs)
            unknown_song_mfcc = songs{s};
            
            % Calculate likelihood for all 3 models
            scores = zeros(1, length(genres));
            for k = 1:length(genres)
                g_name = genres{k};
                scores(k) = get_likelihood(unknown_song_mfcc, models.(g_name));
            end
            
            % Find max score
            [~, max_idx] = max(scores);
            predicted_genre = genres{max_idx};
            fprintf('%-12s | %-12s | %.0f  /  %.0f  /  %.0f\n', ...
                true_genre, predicted_genre, scores(1), scores(2), scores(3));
            
            if strcmp(true_genre, predicted_genre)
                correct = correct + 1;
            end
            total = total + 1;
        end
    end
    
    accuracy = (correct / total) * 100;
    fprintf('\n--> Accuracy for M=%d: %.2f%%\n', M, accuracy);
end

%>>>>>>>HELPER FUNCS<<<<<<<

% --- 1. FEATURE EXTRACTION HELPER ---
function features = extract_genre_features(folder_path, mode)
    files = dir(fullfile(folder_path, '*.wav')); 
    
    if strcmp(mode, 'train')
        features = []; % Stack for training
    else
        features = {}; % Keep separate for testing
    end
    
    for k = 1:length(files)
        file_path = fullfile(folder_path, files(k).name);
        [y, fs] = audioread(file_path);
        
        % MFCC Parameters 
        win_len = round(0.020 * fs);   % 20ms Window
        overlap = round(0.015 * fs);   % 15ms Overlap (5ms Step)
        win_vec = hamming(win_len, 'periodic');
        
            coeffs = mfcc(y, fs, 'Window', win_vec, 'OverlapLength', overlap);
            
            %  Remove 1st coefficient (Energy) 
            coeffs(:, 1) = []; 
            
            %  Cepstral Mean Subtraction (CMS) 
            coeffs = coeffs - mean(coeffs, 1);
            
            % Remove NaNs if any
            coeffs = coeffs(~any(isnan(coeffs), 2), :); 
            
            if strcmp(mode, 'train')
                features = [features; coeffs];
            else
                features{end+1} = coeffs;
            end
        
    end
end

% ---2. GMM TRAINING (EM ALGORITHM)--- 
function [w, mu, sigma] = train_gmm(data, M)
    [N, D] = size(data);
    max_iter = 50; 
    
    % INITIALIZATION 
    try
        [idx, mu] = kmeans(data, M, 'MaxIter', 100, 'Replicates', 1);
    catch
        % Fallback if K-Means fails 
        rand_indices = randperm(N, M);
        mu = data(rand_indices, :);
        idx = randi(M, N, 1);
    end
    
    sigma = zeros(M, D);
    w = zeros(1, M);
    
    % Initial Estimates
    for k = 1:M
        cluster_data = data(idx == k, :);
        w(k) = size(cluster_data, 1) / N;
        if size(cluster_data, 1) > 1
            sigma(k, :) = var(cluster_data) + 1e-4; % Variance Flooring 
        else
            sigma(k, :) = var(data) + 1e-4;
        end
    end
    w = w / sum(w);
    
    % --- EM ITERATIONS ---
    for iter = 1:max_iter
        
        % E-STEP 
        numerator = zeros(N, M);
        for k = 1:M
            diff = data - mu(k, :);
            sigma(k, sigma(k,:) < 1e-6) = 1e-6; % Safety floor
            
            exponent = sum((diff.^2) ./ sigma(k, :), 2);
            coeff = 1 / ((2*pi)^(D/2) * sqrt(prod(sigma(k, :))));
            numerator(:, k) = w(k) * (coeff .* exp(-0.5 * exponent));
        end
        
        denominator = sum(numerator, 2);
        denominator(denominator < realmin) = realmin;
        posteriors = numerator ./ denominator; 
        
        % M-STEP 
        N_k = sum(posteriors, 1) + 1e-10; 
        w = N_k / N;
        mu = (posteriors' * data) ./ N_k';
        sigma = ((posteriors' * (data.^2)) ./ N_k') - mu.^2;
        sigma(sigma < 1e-4) = 1e-4; % Re-apply floor
    end
end

% --- 3. LIKELIHOOD CALCULATION (For Testing) ---
function log_prob = get_likelihood(data, model)
    [T, D] = size(data);
    w = model.w; mu = model.mu; sigma = model.sigma;
    M = length(w);
    frame_prob = zeros(T, 1);
    
    for k = 1:M
        diff = data - mu(k, :);
        exponent = sum((diff.^2) ./ sigma(k, :), 2);
        log_coeff = -0.5 * (D * log(2*pi) + sum(log(sigma(k, :))));
        log_pdf = log_coeff - 0.5 * exponent;
        frame_prob = frame_prob + w(k) * exp(log_pdf);
    end
    
    frame_prob(frame_prob < realmin) = realmin;
    log_prob = sum(log(frame_prob)); 
end