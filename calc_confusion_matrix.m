function confusion_matrix = calc_confusion_matrix(posterior, posterior_index, evidence_vector, variable_index)

    TP = 0;
    FP = 0;
    FN = 0;
    TN = 0;

    if (posterior(posterior_index) > 0.5 && evidence_vector(variable_index) == 2)
        TP = TP + 1;
    end
    if (posterior(posterior_index) > 0.5 && evidence_vector(variable_index) == 1)
        FP = FP + 1;
    end
    if (posterior(posterior_index) < 0.5 && evidence_vector(variable_index) == 2)
        FN = FN + 1;
    end
    if (posterior(posterior_index) < 0.5 && evidence_vector(variable_index) == 1)
        TN = TN + 1;
    end
    
    confusion_matrix = [TP FP FN TN];
end
