function Data = getNonLinearity(Data,nonLin)    
    switch nonLin
        case ''
            Data = rootExpandKernelMap(Data);
        case 'tanh'
            Data = tanh(Data);
        case 'ssr'
            Data = sign(Data).*sqrt(abs(Data));
        case 'chi2'
            Data = vl_homkermap(Data',2,'kchi2');
            Data = Data';
        case 'chi1'
            Data = vl_homkermap(Data',1,'kchi2');
            Data = Data';
        case 'none'
            
        case 'chi2exp'
            u = vl_homkermap(Data',1,'kchi2')';	
            Data = rootExpandKernelMap(u);
        case 'ser'
            Data = rootExpandKernelMap(Data); 
        case 'chi3'
            Data = vl_homkermap(Data',3,'kchi2');
            Data = Data';
        case 'psng'
            Data = PosNegFmap(Data);
            
    end
end

function o = PosNegFmap(x)
    s = sign(x);
    y = sqrt(s.*x);
    o = [y.*(s == 1) y.*(s == -1)];
end