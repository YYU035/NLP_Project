"""Predeclared prediction-only guards; never inspect claim gold at decision time."""
POLICIES=('baseline','bounded_context','protect_contradictions','protect_decisive')

def chosen_arm(policy,focused_prediction):
    if policy not in POLICIES:raise ValueError('Unknown policy')
    if focused_prediction not in ('SUPPORTED','CONTRADICTED','INSUFFICIENT','CONFLICTING'):raise ValueError('Invalid focused verdict')
    if policy=='baseline':return 'baseline'
    if policy=='protect_contradictions' and focused_prediction=='CONTRADICTED':return 'baseline'
    if policy=='protect_decisive' and focused_prediction in ('SUPPORTED','CONTRADICTED'):return 'baseline'
    return 'bounded_abstract'
