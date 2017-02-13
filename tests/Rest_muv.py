'''
Test that the MUV example runs.  Does not yet test that the output is correct.
Only tests MUV-466.
'''
import os, stat, subprocess, shlex, shutil, tempfile

def test_muv():
    '''
    Run the MUV example with a single target
    '''
    muv_dir = os.path.join(os.path.dirname(__file__), '..', 'example', 'muv')
    muv_run_dir = os.path.join(muv_dir, 'muv_run')

    script = os.path.join(muv_run_dir, 'muv_xgboost_models.sh')
    # Make file executable per http://stackoverflow.com/questions/12791997/how-do-you-do-a-simple-chmod-x-from-within-python
    script_st = os.stat(script)
    os.chmod(script, script_st.st_mode | stat.S_IEXEC)

    # Use a temporary directory for output
    result_dir = tempfile.mkdtemp()

    # Create a smaller list of targets to test so the test runs quickly
    # Could add a second target here
    target = 'MUV-466'
    target_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
    with target_file as target_f:
        target_f.write('{}\n'.format(target))

    # Will call the script from muv_run_dir, so make muv_dir relative to it
    muv_dir = os.path.relpath(muv_dir, muv_run_dir)
    command = './{} {} {} {}'.format(os.path.basename(script), muv_dir, result_dir, target_file.name)
    print 'Running MUV test command: {}'.format(command)

    # Run the MUV example script
    os.chdir(muv_run_dir)
    return_code = subprocess.call(shlex.split(command), shell=False)
    assert return_code == 0, 'Non-zero return code' 

    # Can check the new output files with the stored output files here
    # Use filecmp.cmpfiles for exact matches or a custom file comparison
    # for approximate matches
    # For now, only check whether the expected output files were written
    assert os.path.exists(os.path.join(result_dir, 'each_target_cv_result', 'muv_' + target + "_cv_result.csv"))
    assert os.path.exists(os.path.join(result_dir, 'each_target_test_result', 'muv_' + target + "_test_result.csv"))
    assert os.path.exists(os.path.join(result_dir, 'process_time.txt'))
    assert os.path.exists(os.path.join(result_dir, 'muv_cv_result.csv'))
    assert os.path.exists(os.path.join(result_dir, 'muv_test_result.csv'))

    os.remove(target_file.name)
    shutil.rmtree(result_dir)
