"""
Unit-test for defined_eval
"""
from lightchem.eval import defined_eval

def test_defined_eval():
    defined_eval = defined_eval.definedEvaluation()
    assert defined_eval.is_maximize('ROCAUC') == True
    assert defined_eval.stopping_round('ROCAUC') == 100
    try:
        mark = 0
        defind_eval.is_maximize('not_exist_eval_name')
        assert mark == 1
    except ValueError:
        mark = 1
