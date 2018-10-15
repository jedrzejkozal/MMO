rand('state', 0);
randn('state', 0);
Nfindings = 5;
Ndiseases = 3;

N=Nfindings+Ndiseases;
findings = Ndiseases+1:N;
diseases = 1:Ndiseases;
hepatitis_data = load_hepatitis_data()

%generationg graph structure
G = zeros(Ndiseases, Nfindings);
for i=1:Nfindings
  v= rand(1,Ndiseases);
  rents = find(v<0.8);
  if (length(rents)==0)
    rents=ceil(rand(1)*Ndiseases);
  end
  G(rents,i)=1;
end   

prior = [0.2 0.5 0.3];
%leak = 0.5*rand(1,Nfindings); % in real QMR, leak approx exp(-0.02) = 0.98     
leak = 0.98 *ones(1,Nfindings);
inhibit = zeros(Ndiseases, Nfindings);

inhibit(not(G)) = 1;

% first half of findings are +ve, second half -ve
% The very first and last findings are hidden
pos = 1:floor(Nfindings/2);
neg = (pos(end)+1):(Nfindings);

% Make the bnet in the straightforward way
tabular_leaves = 1;
obs_nodes = myunion(pos, neg) + Ndiseases;
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