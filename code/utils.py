# CS505: NLP - Spring 2026

def calculate_accuracy(predictions, labels):
    if len(predictions) == 0:
        return 0.0
    
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    return correct / len(predictions)

def macro_f1(predictions, labels, num_classes=4):
    # TODO: implement the macro-F1 score.
    # Recall that this involves computing the F1 score separately for
    # each label, and then taking the macroaverage. Return the macro-F1
    # score as a floating-point number.
    # STUDENT START --------------------------------------
    f1_scores = []
    
    for label in range(num_classes):
        tp = 0
        fp = 0
        fn = 0
        
        for p, l in zip(predictions, labels):
            if p == label and l == label:
                tp += 1
            elif p == label and l != label:
                fp += 1
            elif p != label and l == label:
                fn += 1
        
        if (2 * tp + fp + fn) > 0:
            f1 = (2 * tp) / (2 * tp + fp + fn)
        else:
            f1 = 0.0
        f1_scores.append(f1)
        
    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    # STUDENT END -------------------------------------------