function W = passNetwork(data,net,isNotsaveSequences)  
    if nargin < 3
        isNotsaveSequences = 1;
    end    
    if size(net,2) == 1
        switch net{1}.poolType
            case 'classical'
               W = genRepresentation(data,net{1}.CVAL,net{1}.nonlinear);
            case 'normalFov'
               W = genFowRepresentation(data,net{1}.CVAL,net{1}.nonlinear); 			
            case 'svr'
               W = liblinearsvr(data,net{1}.CVAL,2); 			   
        end   
        return
    end
    nFrms = size(data,1);
    LIMIT  = net{1}.Window_size;
    if nFrms < LIMIT
        LIMIT = LIMIT * 2;
        nFrms = size(data,1);
        xqv = 1:nFrms/LIMIT:nFrms;
        data = interp1(1:nFrms,data,xqv);
    end

    for layer = 1 : size(net,2)-1
        Window_size = net{layer}.Window_size;
        Stride      = net{layer}.Stride;
        poolType    = net{layer}.poolType;
        normalization = net{layer}.normalization;
        nonlinear = net{layer}.nonlinear;
        CVAL = net{layer}.CVAL;
        if layer == 1
            net{layer}.data = one_layer_darwin(data,Window_size,Stride,poolType,normalization,CVAL,nonlinear);
        else
            net{layer}.data = one_layer_darwin(net{layer-1}.data,Window_size,Stride,poolType,normalization,CVAL,nonlinear);
        end        
    end
    CVAL = net{layer}.CVAL;   
    if isNotsaveSequences == 1
        switch net{end}.poolType
            case 'classical'
                W = genRepresentation(net{end-1}.data,CVAL,net{end}.nonlinear);
            case 'normalFov'
               W = genFowRepresentation(net{end-1}.data,CVAL,net{end}.nonlinear); 			
            case 'svr'
               W = liblinearsvr(net{end-1}.data,CVAL,2);       		
        end   
    else
        W = net{end-1}.data;
    end
end


function out = one_layer_darwin(data,Window_size,Stride,poolType,normalization,CVAL,nonlinear)
    n = size(data,1);
    switch nonlinear
        case 'chi1'
            Dim = size(data,2)*3;
        case 'ser'
            Dim = size(data,2)*2;
        case 'none'
            Dim = size(data,2)*1;
        case 'ssr'
            Dim = size(data,2)*1;
    end
    fstart = 1:Stride:(n-Window_size+1);
    fend = fstart + Window_size -1;    
    switch poolType
        case 'classical'
            out = zeros(numel(fstart),Dim*2);
        case 'normalFov'   
            out = zeros(numel(fstart),Dim);
         case 'svr'    
            out = zeros(numel(fstart),Dim); 
         case 'max'    
            out = zeros(numel(fstart),Dim);    
          
    end
    %parfor chunk = 1 : numel(fstart)
    for chunk = 1 : numel(fstart)
        data_chuck = data(fstart(chunk):fend(chunk),:);
        switch poolType
            case 'classical'
                out(chunk,:) = genRepresentation(data_chuck,CVAL,nonlinear);
            case 'normalFov'
               out(chunk,:) = genFowRepresentation(data_chuck,CVAL,nonlinear); 			
            case 'svr'
               out(chunk,:) = liblinearsvr(data_chuck,CVAL,2); 
            case 'max'
               out(chunk,:) = max(data_chuck);         
        end              
    end
    %fprintf('\n');
    switch normalization
            case 'L2'
                out = normalizeL2(out);
            case 'None'
                
            case 'rootL2'                
                out = normalizeL2(sqrt(out));
            case 'SSRL2'                
                out = normalizeL2(sign(out).*sqrt(abs(out)));    
    end
end

