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

do_experiment(traning_data, validation_data, N, Nfindings, Ndiseases, diseases);