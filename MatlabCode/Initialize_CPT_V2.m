function [CPTList, NoiseList] = Initialize_CPT_V2(NodeTable,FactorMatrix,VSN,VEN,LevelNum)

% IniCPTGenerate

N = length(NodeTable);
CPTList = cell(1,1);
NoiseList = cell(1,1);

for i = 1:(VSN-1)   % parameter
    ParentSum = sum(NodeTable(:,i));
    RandN = 2^ParentSum;
    TempVect = zeros(1,RandN);
    %TempVect2 = zeros(1,RandN);

    if ParentSum == 0
        TempVect = FactorMatrix(i,i);
        %Noise = min(max((randn(1)*0.1+1),0.5),1.5);
        Noise = 1;
        TempVect = min(max(TempVect*Noise,0.000001),0.999999); 
            
        TempVect2 = 1-TempVect;
        TempVect3 = [TempVect TempVect2];
        NoiseVect = Noise;
    else
        NoiseVect = zeros(1,RandN);
        
        % get all factors
        TempCol = FactorMatrix(:,i);
        TempCol(i) = 0; % remove independent factor
        FactorInflu = TempCol;
        FactorInflu(FactorInflu==0) = [];

        DecisionSet = DecisionGenerate(ParentSum,LevelNum);

        for k = 1:RandN
            TempVec = DecisionSet{k,1};
            TempPro = 1-FactorMatrix(i,i); 	% independent factor
            for kk = 1:ParentSum
                if TempVec(kk) == 1         % 1 means false
                    TempPro = TempPro*(1-FactorInflu(kk));
                end
            end
            %Noise = min(max((randn(1)*0.1+1),0.5),1.5);
            Noise = 1;
            TempPro = min(max(TempPro*Noise,0.000001),0.999999);    
            TempVect(k) = 1-TempPro;        % 
            
            NoiseVect(k) = Noise;
        end
        
        TempVect2 = 1-TempVect;
        TempVect3 = [TempVect TempVect2];
    end
    
    CPTList{i,1} = TempVect3;
    NoiseList{i,1} = NoiseVect;
end

for i = VSN:VEN
    ParentSum = sum(NodeTable(:,i));
    RandN = 2^ParentSum;
    
    %--- init vector
    TempVect = ones(1,RandN)/2;
    TempVect(1) = FactorMatrix(1,i);
    TempVect(RandN) = 1-FactorMatrix(i,i);
    
    %NoiseVect = min(max((randn(1,RandN)*0.1+1),0.5),1.5);
    NoiseVect = ones(1,RandN);
    TempVect = TempVect.*NoiseVect;
    TempVect(TempVect > 0.999999) = 0.999999;
    TempVect(TempVect < 0.000001) = 0.000001;
        
    TempVect2 = 1-TempVect;
    TempVect3 = [TempVect TempVect2];
    
    CPTList{i,1} = TempVect3;
    NoiseList{i,1} = NoiseVect;
end

if N > VEN
    for i = (VEN+1):N
        if FactorMatrix(i,i) > 1
            TempVect = zeros(1,4);
            TempVect(4) = 0.96;
            TempVect(3) = TempVect(4)/FactorMatrix(i,i);    %--- FactorMatrix(i,i) the likelihood ratio
            TempVect(2) = 1-TempVect(4);
            TempVect(1) = 1-TempVect(3);
            CPTList{i,1} = TempVect;
        else 
            TempVect = zeros(1,4);
            TempVect(4) = 0.0005;
            TempVect(3) = TempVect(4)/FactorMatrix(i,i);    %--- FactorMatrix(i,i) the likelihood ratio
            TempVect(2) = 1-TempVect(4);
            TempVect(1) = 1-TempVect(3);
            CPTList{i,1} = TempVect;
        end
    end
end