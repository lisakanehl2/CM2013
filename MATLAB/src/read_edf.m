function [hdr, record] = read_edf(filePath)
%% EXAMPLE: Read an EDF file.
%
fprintf('Reading EDF file: %s\n', filePath);
% Placeholder for actual edfread call
% [hdr, record] = edfread(filePath);

fprintf('Reading EDF file: %s\n', filePath);

try
    % Read EDF into a timetable (Time + channels)
    T = edfread(filePath);              % needs a recent MATLAB
    if ~isduration(T.Time), T.Time = seconds(T.Time); end
    T = retime(T,'regular','linear','TimeStep',median(diff(T.Time)));
catch ME
    if exist('sopen','file')==3 || exist('sopen','file')==2
        try
            HDR = sopen(filePath,'r');
            [S, HDR] = sread(HDR);  sclose(HDR);
            record       = S.';                             % [C x N]
            hdr.label    = cellstr(HDR.Label(:));
            hdr.samples  = double(HDR.SampleRate(:)).';
            fprintf('Read with BIOSIG.\n');
            return
        catch ME2
            warning('BIOSIG fallback failed: %s', ME2.message);
        end
    end
    error(['edfread failed. If MATLAB says "Undefined function edfread", ' ...
           'you need a newer MATLAB or install BIOSIG. Original error: ' ME.message]);
end

% Channel names (exclude the Time column)
vars = string(T.Properties.VariableNames);
vars(vars=="Time") = [];

% Estimate sampling rate from the Time vector
t  = T.Time;
dt = seconds(median(diff(t)));
fs = 1/dt;

% Build outputs expected by your other code
hdr.label   = cellstr(vars);               % e.g., {'C3-A2','EOG(L)','EMG'}
hdr.samples = repmat(fs, 1, numel(vars));  % same fs for all if EDF stores uniform timing

% Stack channels row-wise into a numeric matrix [nChannels x nSamples]
record = zeros(numel(vars), height(T));
for i = 1:numel(vars)
    x = T.(vars(i));
    if iscell(x), x = x{1}; end
    record(i,:) = x(:).';
end

end
