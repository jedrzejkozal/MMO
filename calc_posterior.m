function post = calc_posterior(evidence_vector, engine, diseases, Ndiseases, Nfindings)

evidence = cell(1, length(evidence_vector));
evidence(1:Nfindings) = num2cell(evidence_vector(1:Nfindings));

ll = 0;
tic; [e, ll] = enter_evidence(engine, evidence); toc

post = zeros(1, Ndiseases);
for i=diseases(:)'
    m = marginal_nodes(e, i);
    post(1, i-Nfindings) = m.T(2);
end