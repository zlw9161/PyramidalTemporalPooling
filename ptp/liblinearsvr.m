function [w, p] = liblinearsvr(Data,C,normD)
    if normD == 2
        Data = normalizeL2(Data);
    end    
    if normD == 1
        Data = normalizeL1(Data);
    end    
    N = size(Data,1);
    Labels = [1:N]';
    %nc = feature('numCores');
    %model = train(double(Labels), sparse(double(Data)),sprintf('-c %1.6f -s 11 -q -n %d',C,nc) );
    model = train(double(Labels), sparse(double(Data)),sprintf('-c %1.6f -s 11 -q',C) );
    w = model.w';
    p = Data*w;
end
