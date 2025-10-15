function NetData = BNet_generate_4JVP(VSN,VEN,LevelNum,NodeTable,FactorMatrix)


N = VEN + (VSN-1);  %  all parameters, mesaurements, virtual nodes.

%--- initialize CPT
[CPTList, NoiseList] = Initialize_CPT_V2(NodeTable,FactorMatrix,VSN,VEN,LevelNum);

%--- generate BN
discrete_nodes = 1:N;
node_sizes = 2*ones(1,N); 
bnet = mk_bnet(NodeTable, node_sizes, 'discrete', discrete_nodes);

for i=1:N %????????????Temp???????????
        bnet.CPD{i} = tabular_CPD(bnet, i, CPTList{i,1});
end

NetData = struct('Net',bnet,'N',N,'VSN',VSN,'VEN',VEN,'LevelNum',LevelNum,...
    'NodeTable',NodeTable,'FactorMatrix',FactorMatrix,...
    'CPTList',{CPTList},'NoiseList',{NoiseList});

%save date
%cd 'D:\GoogleDrive\PythonScripts\JVS\MatlabCode'
save('MidNet0106_4JVS.mat')