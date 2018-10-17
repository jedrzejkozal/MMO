rand('state', 0);
randn('state', 0);
Nfindings = 6;
Ndiseases = 2;

N=Nfindings+Ndiseases;
findings = 1:Nfindings;
diseases = Nfindings+1:N;

data = load_data();
Ntraning = 119;
Nvalidation = length(data)-Ntraning;

validation_data = data(Ntraning+1:length(data));
traning_data = data(1:Ntraning);

%cross validation
Ntrials = length(data);
result = zeros(Ntrials, 2);

for i = 1:Ntrials
    [result(i,1), result(i,2)] = do_experiment(traning_data, validation_data, N, Nfindings, Ndiseases, diseases);
    
    new_validation_data = [validation_data(2:Nvalidation) traning_data(1)]; %take onother validation set with 20 elements
    traning_data = [traning_data(2:Ntraning) validation_data(1) ];
    validation_data = new_validation_data;
end

u1 = sum(result(:,1))/Ntrials
u2 = sum(result(:,2))/Ntrials