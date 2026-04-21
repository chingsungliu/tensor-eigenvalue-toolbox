rng(42);
A = rand(10, 10);
sigma = 1.5;
B = gaussian_blur(A, sigma);

scriptdir = fileparts(mfilename('fullpath'));
outpath = fullfile(scriptdir, 'reference.mat');
save(outpath, 'A', 'B', 'sigma');

fprintf('reference.mat saved to: %s\n', outpath);
fprintf('A size: %dx%d, sigma: %.4f, B(1,1): %.6f\n', ...
    size(A, 1), size(A, 2), sigma, B(1, 1));
