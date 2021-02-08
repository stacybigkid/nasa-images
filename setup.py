# script to start the apod api
# and activate the relevant venv

import os 

os.chdir('apod-api/')
os.system('python application.py')
os.system('cd ..')
