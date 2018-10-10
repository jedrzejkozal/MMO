clear all;

G = ones(2,3);
G(2,3) = 0;
inhibit = [0.0 0.0 1.0; 0.0 0.0 0.0];
leak = [0.9 0.9 0.9];
prior = [0.4 0.6];

bnet = mk_qmr_bnet(G, inhibit, leak, prior);

engine = jtree_inf_engine(bnet);

N = 5;
evidence = cell(1,N);
evidence{5} = 2;

[engine, loglik] = enter_evidence(engine, evidence);

marg = marginal_nodes(engine, 1);
marg.T 
p = marg.T(2) 