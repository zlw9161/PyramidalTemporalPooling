% *************************************************************************
% Dependency : Lib Linear
% Function : define multi-layer temporal pooling network 
% *************************************************************************
function net = defineNetwork(CVAL,WIN,STRIDE)     
    
    net = cell(1,1);
    
    layer = 1;
    net{layer}.Window_size = WIN;%WIN
    net{layer}.Stride = STRIDE;
    net{layer}.poolType = 'classical';  %options : svr, normalFov, classical
    net{layer}.normalization = 'L2';  %options : None, L2, SSRL2, rootL2
    net{layer}.nonlinear = 'ser'; % The default non-linear function is SER
    net{layer}.CVAL = CVAL;
    
    % Here you can add more layers. Just copy layer 1 and change the ID.
    
%     layer = layer + 1;
%     net{layer}.Window_size = WIN;
%     net{layer}.Stride = STRIDE;
%     net{layer}.poolType = 'classical';  %options : svr, normalFov, classical
%     net{layer}.normalization = 'L2';  %options : None, L2, SSRL2, rootL2
%     net{layer}.nonlinear = ''; % The default non-linear function is SER
%     net{layer}.CVAL = CVAL;
    
    % The final rank pooling layer
    layer = layer + 1;
    net{layer}.Window_size = -1;
    net{layer}.Stride = -1;
    net{layer}.poolType = 'classical'; %svr normalFov classical
    net{layer}.normalization = 'L2'; % None L2
    net{layer}.nonlinear = 'ser'; % The default non-linear function is SER
    net{layer}.CVAL = CVAL;
    
    

    
end
