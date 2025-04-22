"""
    Main file for the project.  
    
    This file is used to run the project and test the CNN.

"""

import sys
from src.training import training
from src.prediction import predict

def run(program_to_run):
    """
        Run the project.
    """
 
    if program_to_run == 'training':
        training()
    elif program_to_run == 'predict':
        predict()

if __name__=='__main__':
    if len(sys.argv) > 1 and sys.argv[1] in ['training', 'predict']:
        run(sys.argv[1])
    else:
        print("Invalid command. Please use one of the following commands:")
        print("training","predict")
