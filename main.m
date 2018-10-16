rand('state', 0);
randn('state', 0);
Nfindings = 6;
Ndiseases = 2;

N=Nfindings+Ndiseases;
findings = Ndiseases+1:N;
diseases = 1:Ndiseases;

data = load_data();
Ntraning = 100;
Nvalidation = length(data)-Ntraning;
traning_data = data(1:Ntraning);
validation_data = data(Ntraning+1:length(data));

%generationg graph structure
G = ones(Ndiseases, Nfindings);

prior = calc_prior(traning_data, Nfindings, Ndiseases);
leak = 0.98 *ones(1,Nfindings); % in real QMR, leak approx exp(-0.02) = 0.98 
inhibit = zeros(Ndiseases, Nfindings);

inhibit(not(G)) = 1;

% first half of findings are +ve, second half -ve
% The very first and last findings are hidden
pos = 1:floor(Nfindings/2);
neg = (pos(end)+1):(Nfindings);
obs_nodes = myunion(pos, neg) + Ndiseases;

% Make the bnet in the straightforward way
tabular_leaves = 1;
bnet = mk_qmr_bnet(G, inhibit, leak, prior, tabular_leaves, obs_nodes);
evidence = cell(1, N);
evidence(findings(pos)) = num2cell(repmat(2, 1, length(pos)));
evidence(findings(neg)) = num2cell(repmat(1, 1, length(neg)));

engine = jtree_inf_engine(bnet);
ll = 0;

tic; [engine, ll] = enter_evidence(engine, evidence); toc

post = zeros(1, Ndiseases);
for i=diseases(:)'
    m = marginal_nodes(engine, i);
    post(1, i) = m.T(2);
end

post