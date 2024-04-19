import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

'''
    Parse through confusion_matrix.txt
    Plot the confusion matrix
'''
if __name__ == '__main__':
    '''
        Parse through confusion_matrix.txt
    '''
    with open('confusion_matrix2.txt', 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    
    # only keep lines that start with Matrix
    lines = [line for line in lines if line.startswith('Matrix:')]
    
    '''
        Line format:
        Matrix: i j conf where i and j are row and column indices and conf is the confusion value
    '''
    
    confusion = np.zeros((1000, 1000))
    for line in lines:
        line = line.split()
        i = int(line[1])
        j = int(line[2])
        conf = float(line[3])
        confusion[i][j] = conf
    
    threshold = 0.9
    mask = confusion > threshold
    confusion[mask] = 1
    confusion[~mask] = 0
    
    '''
        Plot the confusion matrix as a heatmap
    '''
    plt.imshow(confusion, interpolation='nearest')
    plt.colorbar().set_label('Confusion Value')
    plt.title(f'Confusion Matrix > {threshold}')
    
    plt.show()