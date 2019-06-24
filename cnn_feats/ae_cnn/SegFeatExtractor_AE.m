function SegFeatExtractor_AE

featdir = 'D:/Experiments/cnn_feats/ae_cnn/feats/test_ovlp_spct.mat';
labeldir = 'D:/Experiments/cnn_feats/ae_cnn/feats/test_ovlp_label.mat';
datalistdir = 'D:/Experiments/data/AudioEventDataset/lists/test_audio.txt';
labellistdir = 'D:/Experiments/data/AudioEventDataset/lists/test_label.txt';
datalist = importdata(datalistdir);
labellist = importdata(labellistdir);
datanum = size(datalist,1);
label = zeros(datanum,1);
disp(datanum);
seglen = 16480; % time length 100
%seglen = 16320; % time length 50
seg_shift = 16000*0.5;
isovlp = 1;
window = 640;
frm_shift = 160; % time length 100
%frm_shift = 320; % time length 50
patch_size = floor((seglen - (window - frm_shift)) / frm_shift);
feat_dim = 129;% mfcc:150; spctgrm: nfft/2+1; logmel: 120;
[segnum, totalsegs] = cal_segnum(datalist,seglen,seg_shift,isovlp);
if isovlp == 1
    feats = zeros(patch_size, feat_dim, totalsegs, 'single');
else
    feats = zeros(patch_size, feat_dim, totalsegs, 'single');
    seg_shift = seglen;
end
k = 0;
for i = 1 : datanum
    [x,Fs] = audioread(datalist{i});
    if size(x,1) < (seglen * 3)
        x = cat(1, x, x);
    else
        x = cat(1, x, x((size(x,1)-seglen+1):end,:));
    end
    for j = 1 : segnum(i)
        if isovlp == 0
            head = 1 + (j-1) * seglen; 
            tail = j * seglen;
            %tmp = melcepst(x(head:tail), Fs, 'E0dD', 24, 29, window, frm_shift, 0, 0.5);
            tmp = spectrogram(x(head:tail), window, window-frm_shift, 256, Fs)';
            %tmp = logmelbanks(x(head:tail), Fs, 'p0', feat_dim, window, frm_shift, 0, 0.5);
            % for log mel feats
            %tmp(:,end) = [];
            feats(1:size(tmp,1), :, k+j) = tmp;
        else
            head = 1 + (j-1) * seg_shift;
            tail = (j-1) * seg_shift + seglen;
            %tmp = melcepst(x(head:tail), Fs, 'E0dD', 24, 29, window, frm_shift, 0, 0.5);
            tmp = spectrogram(x(head:tail), window, window-frm_shift, 256, Fs)';
            %tmp = logmelbanks(x(head:tail), Fs, 'p0', feat_dim, window, frm_shift, 0, 0.5);
            % for log mel feats
            %tmp(:,end) = [];
            feats(1:size(tmp,1), :, k+j) = single(tmp);
        end
        scene = labellist{i};
        switch scene 
            case 'acoustic_guitar'
                label(k+j) = 1;
            case 'airplane'
                label(k+j) = 2;
            case 'applause'
                label(k+j) = 3;
            case 'bird'
                label(k+j) = 4;
            case 'car'
                label(k+j) = 5;
            case 'cat'
                label(k+j) = 6;
            case 'child'
                label(k+j) = 7;
            case 'church_bell'
                label(k+j) = 8;
            case 'crowd'
                label(k+j) = 9;
            case 'dog_barking'
                label(k+j) = 10;
            case 'engine'
                label(k+j) = 11;
            case 'fireworks'
                label(k+j) = 12;
            case 'footstep'
                label(k+j) = 13;
            case 'glass_breaking'
                label(k+j) = 14;
            case 'hammer'
                label(k+j) = 15;
            case 'helicopter'
                label(k+j) = 16;
            case 'knock'
                label(k+j) = 17;
            case 'laughter'
                label(k+j) = 18;
            case 'mouse_click'
                label(k+j) = 19;
            case 'ocean_surf'
                label(k+j) = 20;
            case 'rustle'
                label(k+j) = 21;
            case 'scream'
                label(k+j) = 22;
            case 'speech_fs'
                label(k+j) = 23;
            case 'squeak'
                label(k+j) = 24;
            case 'tone'
                label(k+j) = 25;
            case 'violin'
                label(k+j) = 26;
            case 'water_tap'
                label(k+j) = 27;
            case 'whistle'
                label(k+j) = 28;
        end
    end
    k = k + segnum(i);
    disp(i);
end
disp(totalsegs);
% disp(k);
labels = label';
save(labeldir,'labels');
save(featdir,'feats');
fprintf('Complete...\n');
end

% -------------------------------------------------------------------------
function [segnum, totalsegs] = cal_segnum(datalist,seglen,shift,isovlp)
% -------------------------------------------------------------------------
datanum = size(datalist, 1);
segnum = zeros(datanum, 1);
for k = 1 : datanum
    [x, ~] = audioread(datalist{k});
    if size(x,1) < (seglen * 3)
        x = cat(1, x, x);
    else
        x = cat(1, x, x((size(x,1)-seglen+1):end,:));
    end
    audiolen = size(x,1);
    if isovlp == 0
        segnum(k) = floor(audiolen / seglen);
    else
        if (audiolen-(floor(audiolen/shift)-1)*shift) < (seglen-shift) 
            segnum(k) = floor(audiolen / shift) - 2;
        else
            segnum(k) = floor(audiolen / shift) - 1;
        end
    end 
end
totalsegs = sum(segnum);
end