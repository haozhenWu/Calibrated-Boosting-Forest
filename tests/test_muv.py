'''
Test that the MUV example runs.  Does not yet test that the output is correct.
'''
import os, stat, subprocess, shlex, shutil, tempfile

def test_muv():
    '''
    Run the MUV example
    '''
    muv_dir = os.path.join(os.path.dirname(__file__), '..', 'example', 'muv')
    muv_run_dir = os.path.join(muv_dir, 'muv_run')

    script = os.path.join(muv_run_dir, 'muv_xgboost_models.sh')
    # Make file executable per http://stackoverflow.com/questions/12791997/how-do-you-do-a-simple-chmod-x-from-within-python
    script_st = os.stat(script)
    os.chmod(script, script_st.st_mode | stat.S_IEXEC)

    # Use a temporary directory for output
    result_dir = tempfile.mkdtemp()

    # Will call the script from muv_run_dir, so make muv_dir relative to it
    muv_dir = os.path.relpath(muv_dir, muv_run_dir)
    command = './{} {} {} muv_TargetName.csv'.format(os.path.basename(script), muv_dir, result_dir)
    print 'Running MUV test command: {}'.format(command)

    # Run the MUV example script
    os.chdir(muv_run_dir)
    subprocess.call(shlex.split(command), shell=False)

    # Can check the new output files with the stored output files here
    # Use filecmp.cmpfiles for exact matches or a custom file comparison
    # for approximate matches

    shutil.rmtree(result_dir)
