function B = gaussian_blur(A, sigma)
    x = -ceil(3*sigma):ceil(3*sigma);
    h = exp(-x.^2 / (2*sigma^2));
    h = h / sum(h);
    B = conv2(h, h, A, 'same');
end
