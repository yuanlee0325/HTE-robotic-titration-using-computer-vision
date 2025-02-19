""" SMB network settings """

from smb.SMBConnection import SMBConnection
from smb.smb_structs import OperationFailure
import pandas as pd
import tempfile
import os


# The username for the network share
USERNAME = "xxx"
# The password for the network share
PASSWORD = "xxx"
# The NetBIOS name of the Opentrons robot 
MY_NAME = "xxx"
# The NetBIOS name of the remote server
REMOTE_NAME = "xxx"
# The IP address of the remote server
REMOTE_IP = "xxx"
# 139 for SMB over NetBIOS (what we are doing here)
PORT = 139
# The name of the shared folder on the remote server
SERVICE_NAME = "DriveSMB"


""" SMB network actions """

# Set up the SMB connection
def smb_connect():
    print(MY_NAME)
    print(REMOTE_IP)
    connection = SMBConnection(
        username=USERNAME,
        password=PASSWORD,
        my_name=MY_NAME,
        remote_name=REMOTE_NAME,
        use_ntlm_v2=True
        )

    assert connection.connect(
        ip=REMOTE_IP,
        port=PORT,
        timeout=10
        ),'communication not established'

    return connection

# Store an existing file on disk on the SMB server
def smb_store_file(connection,
                   remote_path='/test_file.txt', 
                   local_path='/var/lib/jupyter/notebooks/test.txt'):
    '''
    args:
    - remote_path : add filename to be saved in the remote path
    - local_path : local path including filename
    
    '''

    tmp = open(local_path, "rb")

    connection.storeFile(
        SERVICE_NAME,
        remote_path,
        tmp,
        timeout=5
        )

    # Clean up
    tmp.close()
    
    #connection.close()

# Retrieve a file from the SMB server
def smb_retrieve_file(connection,
                      remote_path='/test_file.txt', 
                      local_path='/var/lib/jupyter/notebooks/msi2py.csv'):

    # Get the file from the server
    tmp = tempfile.NamedTemporaryFile()
    file_attributes, filesize = connection.retrieveFile(
        service_name=SERVICE_NAME,
        path=remote_path,
        file_obj=tmp
        )

    # Write the file to the local disk
    outfile = open(local_path, "wb")
    # rewind buffer
    tmp.seek(0)
    # You could also directly use the contents here
    contents = tmp.read()
    outfile.write(contents)

    # Clean up
    tmp.close()
    outfile.close()
    
    #connection.close()
    
    
    
def save_trigger_file(fname="pi2msi.txt",
                      content='I am done!'):
    # Saving file for msi
    file = open(fname, "w+")
    file.write(content)
    file.close()

def is_file_exist(connection,
                  fname):
    #connection = smb_connect()
    file_attr=None
    
    try:
        file_attr=connection.getAttributes(SERVICE_NAME,fname)
    except OperationFailure:
            print("Oops!  file does not exist")
           
    return file_attr

def is_csv_file_and_status(path, fname = 'msi2pi_copy.csv'):
    isFile = None
    isRun = None
    
    # check file copied from msi
    path=os.path.join(path,fname)
    isFile = os.path.isfile(path)
    df=pd.read_csv(path);
    isRun = True if df.status[0] == 'run' else False
    
    print('file exists? : {}'.format(isFile))
    print('run robot? : {}'.format(isRun))
    
    return (isFile and isRun)