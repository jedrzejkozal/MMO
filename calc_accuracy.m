function [Acc1, Acc2] = calc_accuracy(validation_data, engine, diseases, Ndiseases, Nfindings)

confusion_matrix1 = zeros(1,4); % TP FP FN TN
confusion_matrix2 = zeros(1,4);

for i = 1:length(validation_data)
    evidence_vector = validation_data{i};
    post = calc_posterior(evidence_vector, engine, diseases, Ndiseases, Nfindings);
    
    confusion_matrix1 = confusion_matrix1 + calc_confusion_matrix(post, 1, evidence_vector, Nfindings+1);
    confusion_matrix2 = confusion_matrix2 + calc_confusion_matrix(post, 2, evidence_vector, Nfindings+2);
end

TPR1 = confusion_matrix1(1)/(confusion_matrix1(1)+confusion_matrix1(3));
FPR1 = confusion_matrix1(2)/(confusion_matrix1(2)+confusion_matrix1(4));
FNR1 = confusion_matrix1(3)/(confusion_matrix1(1)+confusion_matrix1(3));
TNR1 = confusion_matrix1(4)/(confusion_matrix1(2)+confusion_matrix1(4));

LR_plus1 = TPR1/FPR1;
LR_minus1 = FNR1/TNR1;

Acc1 = (confusion_matrix1(1) + confusion_matrix1(4))/(confusion_matrix1(1) + confusion_matrix1(2) + confusion_matrix1(3) + confusion_matrix1(4));

TPR2 = confusion_matrix2(1)/(confusion_matrix2(1)+confusion_matrix2(3));
FPR2 = confusion_matrix2(2)/(confusion_matrix2(2)+confusion_matrix2(4));
FNR2 = confusion_matrix2(3)/(confusion_matrix2(1)+confusion_matrix2(3));
TNR2 = confusion_matrix2(4)/(confusion_matrix2(2)+confusion_matrix2(4));

LR_plus2 = TPR2/FPR2;
LR_minus2 = FNR2/TNR2;

Acc2 = (confusion_matrix2(1) + confusion_matrix2(4))/(confusion_matrix2(1) + confusion_matrix2(2) + confusion_matrix2(3) + confusion_matrix2(4));

confusion_matrix1
LR_plus1;
LR_minus1;
Acc1
confusion_matrix2
LR_plus2;
LR_minus2;
Acc2

end