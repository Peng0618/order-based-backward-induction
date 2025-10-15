function ProbVector = BNet_inference2(bnet,Evidence)
N = length(Evidence);

ProbVector = zeros(1,N);

%engine = jtree_inf_engine(bnet);
engine = pearl_inf_engine(bnet);
%engine = belprop_inf_engine(bnet);


evidence = cell(1,N);
evidence = Evidence;
[engine, loglik] = enter_evidence(engine, evidence);

for i = 1:N
    marg = marginal_nodes(engine, i);
    ProbVector(i) = marg.T(1);
end
