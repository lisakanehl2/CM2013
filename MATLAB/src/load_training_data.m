function [eeg_data, labels] = load_data(edfFilePath, xmlFilePath)
%% Load EDF (signal) + XML (sleep stages) for ONE recording.
% Outputs:
%   eeg_data : [nEpochs x 3750] double  (30 s @ 125 Hz, single EEG channel)
%   labels   : [1 x nEpochs] in {0,1,2,3,4} = {Wake,N1,N2,N3,REM}

    fprintf('Loading training data from %s and %s...\n', edfFilePath, xmlFilePath);

    % ---- 1) EDF: read all channels ----
    [hdr, record] = read_edf(edfFilePath);
    disp('Channels found in EDF:');
    disp(hdr.label);

    % ---- 2) Pick one EEG channel (prefer C3/C4/EEG) ----
    chanNames = string(hdr.label);
    eegIdx = find(contains(upper(chanNames),"EEG") | contains(upper(chanNames),["C3","C4"]), 1, 'first');
    if isempty(eegIdx), eegIdx = 1; end
    x = double(record(eegIdx, :));      % 1 x N (continuous EEG)

    % ---- 3) Sampling rate ----
    if isfield(hdr,'samples') && numel(hdr.samples) >= eegIdx && ~isempty(hdr.samples(eegIdx))
        fs = double(hdr.samples(eegIdx));
    elseif isfield(hdr,'fs') && ~isempty(hdr.fs)
        fs = double(hdr.fs(1));
    else
        error('Sampling rate not found in EDF header.');
    end

    % ---- 4) Standardize to 125 Hz ----
    fsTarget = 125;
    if abs(fs - fsTarget) > 1e-6
        x = resample(x, fsTarget, round(fs));
        fs = fsTarget;
    end

    % ---- 5) Segment into 30-second epochs ----
    epochLenSec = 30;
    epochLen    = epochLenSec * fs;     % 30 * 125 = 3750
    nEpochs     = floor(numel(x) / epochLen);
    if nEpochs == 0
        error('Recording too short for a 30 s epoch.');
    end
    x = x(1 : nEpochs * epochLen);                    % trim to full epochs
    eeg_data = reshape(x, epochLen, nEpochs).';       % [nEpochs x samples]

    % ---- 6) XML: read per-second stages, convert to per-epoch labels ----
    [~, stagesSec, xmlEpochLen, annotation] = readXML(xmlFilePath);
    if annotation == 0
        error('XML has no annotations.');
    end
    if xmlEpochLen ~= epochLenSec
        warning('XML epoch length = %d s, using mode over %d-s windows.', xmlEpochLen, epochLenSec);
    end

    % readXML() returns per-second codes with this meaning:
    %   5 = Wake, 4 = N1, 3 = N2, 2 = N3, 1 = N4, 0 = REM
    % We map to 0..4 = {Wake,N1,N2,N3,REM} (merge N4 into N3)

    needSec = nEpochs * epochLenSec;
    if numel(stagesSec) < needSec
        stagesSec(end+1:needSec) = stagesSec(end);    % pad with last code
    elseif numel(stagesSec) > needSec
        stagesSec = stagesSec(1:needSec);
    end

    labels = zeros(1, nEpochs);
    for e = 1:nEpochs
        s0 = (e-1)*epochLenSec + 1;
        s1 = e*epochLenSec;
        win = stagesSec(s0:s1);
        m = mode(win);  % most frequent second-level code in this epoch

        switch m
            case 5, labels(e) = 0;        % Wake -> 0
            case 4, labels(e) = 1;        % N1   -> 1
            case 3, labels(e) = 2;        % N2   -> 2
            case {2,1}, labels(e) = 3;    % N3/N4 -> 3
            case 0, labels(e) = 4;        % REM  -> 4
            otherwise
                labels(e) = 0;            % fallback to Wake
        end
    end

    fprintf('Loaded %d epochs (%.1f hours) at %g Hz from channel "%s".\n', ...
        nEpochs, nEpochs*epochLenSec/3600, fs, char(chanNames(eegIdx)));
end
