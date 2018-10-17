rand('state', 0);
randn('state', 0);
Nfindings = 6;
Ndiseases = 2;

N=Nfindings+Ndiseases;
findings = 1:Nfindings;
diseases = Nfindings+1:N;

data = load_data();
Ntraning = 100;
Nvalidation = length(data)-Ntraning;
traning_data = data(1:Ntraning);
validation_data = data(Ntraning+1:length(data));

%generate graph structure
G = ones(Ndiseases, Nfindings);

prior = calc_prior(traning_data, Nfindings, Ndiseases);
leak = 0.98 *ones(1,Nfindings); % in real QMR, leak approx exp(-0.02) = 0.98 
inhibit = calc_inhibit(traning_data, Nfindings, Ndiseases, prior);
inhibit(not(G)) = 1;

obs_nodes = 1:N;

% Make the bnet in the straightforward way
tabular_leaves = 1; %   = 0 means noisy-OR leaves
bnet = mk_qmr_bnet(G, inhibit, leak, prior, tabular_leaves, obs_nodes);

engine = jtree_inf_engine(bnet);

confusion_matrix1, confusion_matrix2, Acc1, Acc2 = calc_accuracy(validation_data, engine, diseases, Ndiseases, Nfindings);