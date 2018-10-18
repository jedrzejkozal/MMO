function result = do_experiment(traning_data, validation_data, N, Nfindings, Ndiseases, diseases)

%generate graph structure
%G = ones(Ndiseases, Nfindings);
G = [ 0 0 0 1 1 1; 1 1 1 1 1 1 ];

prior = calc_prior(traning_data, Nfindings, Ndiseases);
%leak = 0.98 *ones(1,Nfindings); % in real QMR, leak approx exp(-0.02) = 0.98 
leak = 0.98*ones(1,Nfindings);
inhibit = calc_inhibit(traning_data, Nfindings, Ndiseases, prior);
inhibit(not(G)) = 1;

obs_nodes = 1:N;

% Make the bnet in the straightforward way
tabular_leaves = 0; %   = 1 means multinomial leaves (ignores leak/inhibit params), = 0 means noisy-OR leaves
bnet = mk_qmr_bnet(G, inhibit, leak, prior, tabular_leaves, obs_nodes);

engine = jtree_inf_engine(bnet);

Acc1 = 0;
Acc2 = 0;
result = calc_accuracy(validation_data, engine, diseases, Ndiseases, Nfindings);

end