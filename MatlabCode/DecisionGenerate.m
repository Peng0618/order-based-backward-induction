function DecisionSet = DecisionGenerate(Length,LevelNum)

if Length == 0
    f = errordlg('The length is zero!','File Error');
    %f = warndlg('The length is zero!','Warning');
end

DecisionN = LevelNum^Length;
T = Length;

DecisionSet = cell(1,1);
TempD = zeros(1,T);
for i = 1:DecisionN
    NumDec = i-1;
    
    for j = 1:(T-1)
        Deno = LevelNum^(T-j);
        Rem = floor(double(NumDec)/double(Deno));   % revised at 0708/2021: add double
        TempD(j) = 1+Rem;
        NumDec = NumDec-Deno*Rem;
    end
    
    TempD(T) = 1+NumDec;
    
    DecisionSet{i,1} = TempD;
end