function NetData = BNet_generateV1(NodeTable,LevelNum,VSN)

% A: theta1, B: theta2, CDEF: V1-V4

N = length(NodeTable);                                           %%
%VSN = 6;                                        %%
%LevelNum = 2;

for i=1:(VSN-1) %????????????Temp???????????
        eval(['T',num2str(i),'=','i',';']);
end
for i=VSN:N %????????????Temp???????????
        eval(['V',num2str(i),'=','i',';']);
end

dag = NodeTable;                %%

discrete_nodes = 1:N;
node_sizes = LevelNum*ones(1,N); 

bnet = mk_bnet(dag, node_sizes, 'discrete', discrete_nodes);          

CPTList = Initialize_CPT(NodeTable,LevelNum);

for i=1:N %????????????Temp???????????
        bnet.CPD{i} = tabular_CPD(bnet, i, CPTList{i,1});
end

%bnet.CPD{A} = tabular_CPD(bnet, A, Vec1);
%bnet.CPD{B} = tabular_CPD(bnet, B, Vec2);
%bnet.CPD{C} = tabular_CPD(bnet, C, Vec3);
%bnet.CPD{D} = tabular_CPD(bnet, D, [0.9 0.2 0.6 0.1 0.1 0.8 0.4 0.9]);
%bnet.CPD{E} = tabular_CPD(bnet, E, [0.9 0.2 0.1 0.8]);
%bnet.CPD{F} = tabular_CPD(bnet, F, [0.9 0.3 0.4 0.1 0.1 0.7 0.6 0.9]);

% G = bnet.dag;
% draw_graph(bnet.dag)

NetData = struct('Net',bnet,'N',N,'VSN',VSN);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%T = 1;
%ActList = cell(1,T);
%ActList{1,1}=[0 B C];
%ActList{1,2}=[0 C];
%ActList{1,3}=[0 B C];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Cost of Activity
%CA = [0 50 100];

% Cost of Rework
%CR = [100 200 300 400];

% Number of key node and Cost of System Failure
%CF = [1 6000];

%CostData = struct('T',T,'CA',CA,'CR',CR,'CF',CF,'BeliefThres',BeliefThres);

