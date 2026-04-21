function generate_tenpow_reference()
%GENERATE_TENPOW_REFERENCE  Produce bit-level reference data for the
%  Python tenpow port's parity test.
%
%  Reference implementation: verbatim copy of tenpow from HONI.m line 129-139
%  (identical to the definition in Multi.m line 66-76), provided here as a
%  local function since tenpow is not a MATLAB top-level callable in the
%  canonical sources.

    rng(42);
    x = rand(5, 1);

    tp2 = tenpow(x, 2);
    tp3 = tenpow(x, 3);
    tp4 = tenpow(x, 4);

    scriptdir = fileparts(mfilename('fullpath'));
    outpath = fullfile(scriptdir, 'tenpow_reference.mat');
    save(outpath, 'x', 'tp2', 'tp3', 'tp4');

    fprintf('tenpow_reference.mat saved to: %s\n', outpath);
    fprintf('x size: [%dx%d], tp2 len=%d, tp3 len=%d, tp4 len=%d\n', ...
        size(x, 1), size(x, 2), length(tp2), length(tp3), length(tp4));
    fprintf('x(1)   = %.15f\n', x(1));
    fprintf('tp2(1) = %.15f\n', tp2(1));
    fprintf('tp4(end) = %.15f\n', tp4(end));
end


function x_p = tenpow(x, p)
% Verbatim copy from matlab_ref/hni/HONI.m line 129-139.
% (The definition in Multi.m line 66-76 is byte-identical.)
    if p == 0
        x_p = 1;
    else
        x_p = x;
        for i = 1:p-1
            x_p = kron(x, x_p);
        end
    end
end
