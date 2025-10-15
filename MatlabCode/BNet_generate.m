function bnet = BNet_generate()
% A: theta1, B: theta2, CDEF: V1-V4
N = 6; A = 1; B = 2; C = 3; D = 4; E = 5; F = 6;
dag = zeros(N,N);

dag(A,[B C D]) = 1;
dag(B,[E F]) = 1;
dag(C,D)=1;
dag(E,F)=1;

% draw_graph(dag)

% %% 1 good; 2 fail
% CPT = cell(1,N);
% CPT{1,1}=[0.8 0.2];
% CPT{1,2}=[0.8 0.1 0.2 0.9];
% CPT{1,3}=[0.7 0.2 0.3 0.8];
% CPT{1,4}=[0.9 0.4 0.5 0.1 0.1 0.6 0.5 0.9];
% CPT{1,5}=[0.9 0.4 0.1 0.6];
% CPT{1,6}=[0.9 0.3 0.3 0.1 0.1 0.7 0.7 0.9];
% %%

discrete_nodes = 1:N;
node_sizes = 2*ones(1,N); 

bnet = mk_bnet(dag, node_sizes, 'discrete', discrete_nodes);

bnet.CPD{A} = tabular_CPD(bnet, A, [0.2 0.8]);
bnet.CPD{B} = tabular_CPD(bnet, B, [0.8 0.1 0.2 0.9]);
bnet.CPD{C} = tabular_CPD(bnet, C, [0.7 0.2 0.3 0.8]);
bnet.CPD{D} = tabular_CPD(bnet, D, [0.9 0.2 0.6 0.1 0.1 0.8 0.4 0.9]);
bnet.CPD{E} = tabular_CPD(bnet, E, [0.9 0.2 0.1 0.8]);
bnet.CPD{F} = tabular_CPD(bnet, F, [0.9 0.3 0.4 0.1 0.1 0.7 0.6 0.9]);

% G = bnet.dag;
% draw_graph(bnet.dag)




