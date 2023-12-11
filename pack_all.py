import util
import subprocess
import pack
import pdb

S = [0.8, 0.85, 0.9, 0.95, 1.0]

for i, s in enumerate(S):
    # pdb.set_trace()
    util.split_dataset(s)
    subprocess.run('python train.py')
    subprocess.run('python test_script.py')
    pack.pack_model('best' + str(i), input_name='best.pt')