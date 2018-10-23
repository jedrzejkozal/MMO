rand('state', 0);
randn('state', 0);
rand('twister',sum(100*clock));
randn('state',sum(100*clock));

Nfindings = 6;
Ndiseases = 2;

N=Nfindings+Ndiseases;
findings = 1:Nfindings;
diseases = Nfindings+1:N;

%data = load_data();
data = generate_data();

%shuffle data
data = data(randperm(numel(data)));

Ntraning = 100;
Nvalidation = length(data)-Ntraning;

validation_data = data(Ntraning+1:length(data));
traning_data = data(1:Ntraning);

%cross validation
Ntrials = length(data);
result = zeros(Ntrials, 6);

for i = 1:Ntrials
    result(i,:) = do_experiment(traning_data, validation_data, N, Nfindings, Ndiseases, diseases);
    
    new_validation_data = [validation_data(2:Nvalidation) traning_data(1)]; %take another validation set with 20 elements
    traning_data = [traning_data(2:Ntraning) validation_data(1) ];
    validation_data = new_validation_data;
end

Uacc1 = sum(result(:,1))/Ntrials
Uacc2 = sum(result(:,2))/Ntrials
ULRplus1 = sum(result(:,3))/Ntrials;
ULRminus1 = sum(result(:,4))/Ntrials;
ULRplus2 = sum(result(:,5))/Ntrials;
ULRminus2 = sum(result(:,6))/Ntrials;