function sqrt2_newton()
%SQRT2_NEWTON  Newton's method for x^2 - 2 = 0, saving full per-iteration state.
%  Generates sqrt2_newton_reference.mat for the Python per-iteration parity POC.
%
%  Deterministic: starts from x_0 = 1.0, runs 10 iterations.
%  Saved fields:
%    x_history   (11, 1) vector [x_0, x_1, ..., x_10]
%    res_history (11, 1) vector [r_0, r_1, ..., r_10]  where r_n = x_n^2 - 2

    x0 = 1.0;
    n_iter = 10;

    x_history = zeros(n_iter + 1, 1);
    res_history = zeros(n_iter + 1, 1);

    % iteration 0 = initial state
    x_history(1) = x0;
    res_history(1) = x0^2 - 2;

    x = x0;
    for k = 1:n_iter
        x = x - (x^2 - 2) / (2 * x);
        x_history(k + 1) = x;
        res_history(k + 1) = x^2 - 2;
    end

    scriptdir = fileparts(mfilename('fullpath'));
    outpath = fullfile(scriptdir, 'sqrt2_newton_reference.mat');
    save(outpath, 'x_history', 'res_history');

    fprintf('sqrt2_newton_reference.mat saved: %s\n', outpath);
    fprintf('x_10   = %.20f\n', x_history(end));
    fprintf('target = %.20f   (sqrt(2))\n', sqrt(2));
    fprintf('res_10 = %.3e\n', res_history(end));
end
