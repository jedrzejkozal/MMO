clear all;

G = ones(1,1);
inhibit = [1.0];
leak = [0.5];
prior = [0.4];

bnet = mk_qmr_bnet(G, inhibit, leak, prior);

engine = jtree_inf_engine(bnet);

N = 2;
evidence = cell(1,N);
evidence{2} = 2;

[engine, loglik] = enter_evidence(engine, evidence);

marg = marginal_nodes(engine, 1);
marg.T 
p = marg.T(2) 