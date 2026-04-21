% Verbatim from matlab_exercise/gaussian_blur.m (whole file).
% Do not edit — this is kept in sync with the canonical source.

function B = gaussian_blur(A, sigma)
    x = -ceil(3*sigma):ceil(3*sigma);
    h = exp(-x.^2 / (2*sigma^2));
    h = h / sum(h);
    B = conv2(h, h, A, 'same');
end
