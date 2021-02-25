import subprocess as sp
import time
import os

'''
command1 = ['conda.bat', 'activate', 'yolov5']
process = sp.Popen(command1, stdin=sp.PIPE, stdout = sp.PIPE)
'''
#sp.run('conda activate yolov5', shell=True)

#text = 'conda activate yolov5'
#cmd = "echo {}".format(text)
#os.system(cmd)
#sp.run('bash -c "source activate yolov5; python -V"', shell=True)

addr = "rtsp://admin:meta33273327@10.0.13.5/"
command2 = ['python.exe', 'detect.py', '--source', addr]
process = sp.Popen(command2, stdin=sp.PIPE)
