function InitPriorVector = inferenceBN(Evidence)
%addpath(genpathKPM(pwd))

%cd 'C:\Users\xupen\Google Drive\PythonScripts\BayesNet\MatlabCode'
%cd 'D:\GoogleDrive\PythonScripts\JVS\MatlabCode'

%load('NetworkData.mat')

load('MidNet0106_4JVS.mat')



EvidenceVector = cell2mat(Evidence);

Index = find(EvidenceVector > 0);


evidence = cell(1,NetData.N);
for i = 1:length(Index)
    evidence{Index(i)} = EvidenceVector(Index(i));
end

InitPriorVector = BNet_inference2(NetData.Net,evidence);

