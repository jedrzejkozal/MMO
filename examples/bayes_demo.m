N = 4; 
dag = zeros(N,N);
C = 1; S = 2; R = 3; W = 4;
dag(C,[R S]) = 1;
dag(R,W) = 1;
dag(S,W) = 1;

discrete_nodes = 1:N;
node_sizes = 2*ones(1,N);

bnet = mk_bnet(dag, node_sizes, 'names', {'cloudy','S','R','W'}, 'discrete', 1:4);


bnet.CPD{C} = tabular_CPD(bnet, C, [0.5 0.5]);
bnet.CPD{R} = tabular_CPD(bnet, R, [0.8 0.2 0.2 0.8]);
bnet.CPD{S} = tabular_CPD(bnet, S, [0.5 0.9 0.5 0.1]);
bnet.CPD{W} = tabular_CPD(bnet, W, 'CPT', [1 0.1 0.1 0.01 0 0.9 0.9 0.99]);

engine = jtree_inf_engine(bnet);

evidence = cell(1,N);
evidence{W} = 2;

[engine, loglik] = enter_evidence(engine, evidence);

marg = marginal_nodes(engine, S);
marg.T; % vector of probabilities of false and true
p = marg.T(2) % p = P(S=2|W=2)

evidence{R} = 2;
[engine, loglik] = enter_evidence(engine, evidence);
marg = marginal_nodes(engine, S);
p = marg.T(2) % p = P(S=2|W=2,R=2)

% bar(marg.T)

%%% For observed nodes
% evidence = cell(1,N);
% evidence{W} = 2;
% engine = enter_evidence(engine, evidence);
% m = marginal_nodes(engine, W);
% m.T

evidence = cell(1,N);
[engine, ll] = enter_evidence(engine, evidence);
m = marginal_nodes(engine, [S R W]);
m.T

evidence{R} = 2;
[engine, ll] = enter_evidence(engine, evidence);
m = marginal_nodes(engine, [S R W], 1);
m.T
                                  
                                                             
