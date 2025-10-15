function CPTList = Initialize_CPT(NodeTable,LevelNum)

% IniCPTGenerate

N = length(NodeTable);
CPTList = cell(1,1);
for i = 1:N
    TempSum = sum(NodeTable(:,i));
    RandN = 2^TempSum;
    TempVect = zeros(1,RandN);
    
    % random CPT
    if TempSum == 0
        %TempVect(1) = min(max(0.02*randn(1)+0.15,0.02),0.98);
        TempVect(1) = min(max(0.01*randn(1)+0.05,0.02),0.98);  % !
    else
        %FactorInflu = min(max(0.03*randn(1,TempSum)+0.1,0.02),0.98);
        %FactorBase = 0.8/TempSum;
        %FactorInflu = min(max(0.05*randn(1,TempSum)+0.75,0.02),0.98);    % the smaller the base number(0.75), the stronger the relationship between causal factors
        %FactorInflu = min(max(0.03*randn(1,TempSum)+0.75,0.02),0.98);    % the smaller the base number(0.75), the stronger the relationship between causal factors
        
        %FactorInflu = min(max(0.03*randn(1,TempSum)+0.6,0.02),0.98); 
        
        %FactorInflu = min(max(0.03*randn(1,TempSum)+0.55,0.02),0.98);        %0909
        FactorInflu = min(max(0.08*randn(1,TempSum)+0.5,0.02),0.98);        %0929
        
        DecisionSet = DecisionGenerate(TempSum,LevelNum);
        for k = 1:RandN
            TempVec = DecisionSet{k,1};
            TempPro = 0.9;                                     % 0.95
            for kk = 1:TempSum
                if TempVec(kk) == 1                             %noisy-or Step 1    06/22
                    TempPro = TempPro*FactorInflu(kk);
                %else
                %    TempPro = TempPro*(1-FactorInflu(kk));
                end
            end
            Noise = min(max((randn(1)*0.1+1),0.5),1.5);
            TempPro = min(max(TempPro*Noise,0.02),0.98);
            TempVect(k) = 1-TempPro;                            %%noisy-or Step 2               
        end
    end
    
    TempVect2 = 1-TempVect;
    TempVect3 = [TempVect TempVect2];
    CPTList{i,1} = TempVect3;
    
    %TempVect = 2*rand(1,RandN);
    %TempVect = TempVect - fix(TempVect);

    %TempVect2 = 1-TempVect;
    %TempVect3 = [TempVect TempVect2];
    %CPTList{i,1} = TempVect3;
    %CPTList{i,1} = TempVect3(end:-1:1);
end