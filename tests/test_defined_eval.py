"""
Unit-test for defined_eval
"""
from lightchem.eval import defined_eval

def test_defined_eval():
    eval = defined_eval.definedEvaluation()
    assert eval.is_maximize('ROCAUC') == True
    assert eval.stopping_round('ROCAUC') == 100
    mark = 0
    try:
        eval.is_maximize('not_exist_eval_name')
        assert mark == 1
    except ValueError:
        mark = 1
