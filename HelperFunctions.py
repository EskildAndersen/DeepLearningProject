'''
Script with helper functions to save chosen settings, save accuracy and loss of models.
Beneficial when training and validating models to finetune parameters. 

'''

from settings import (
    # DECODER
    EMBEDDING_DIM,
    ATTENTION_DIM,
    DECODER_DIM,
    DECODER_DROP_PROB,
    DECODER_PAD_INDEX,

    # GENERAL
    BATCH_SIZE,
    LEARNING_RATE,
    TEATHER_FORCING_PROB,
    WEIGHT_DECAY,
    OPTIMIZER,
    LR_STEP,
    LOSS_PAD_INDEX,
    NUMBER_OF_ITERATIONS,
)
import os


def saveSettings():
    folder = os.path.join('results', 'settings')
    files = set(os.listdir(folder))
    files = files - {'README.md'}
    
    for i in range(len(files)+1):
        if not f'settings_{i}.txt' in files:
            num = i
            break
    
    filename = f'settings_{num}.txt'
    file = os.path.join(folder, filename)
    
    with open(file, 'w') as f:
        f.write(f'EMBEDDING_DIM : {EMBEDDING_DIM}\n')
        f.write(f'ATTENTION_DIM : {ATTENTION_DIM}\n')
        f.write(f'DECODER_DIM : {DECODER_DIM}\n')
        f.write(f'DECODER_DROP_PROB : {DECODER_DROP_PROB}\n')
        f.write(f'DECODER_PAD_INDEX : {DECODER_PAD_INDEX}\n')
        f.write(f'BATCH_SIZE : {BATCH_SIZE}\n')
        f.write(f'LEARNING_RATE : {LEARNING_RATE}\n')
        f.write(f'OPTIMIZER : {str(OPTIMIZER).split(".")[-2]}\n')
        f.write(f'WEIGHT_DECAY : {WEIGHT_DECAY}\n')
        f.write(f'LR_STEP : {LR_STEP}\n')
        f.write(f'LOSS_PAD_INDEX : {LOSS_PAD_INDEX}\n')
        f.write(f'NUMBER_OF_ITERATIONS : {NUMBER_OF_ITERATIONS}\n')
        f.write(f'TEATHER_FORCING_PROB : {TEATHER_FORCING_PROB}\n')
    
    return filename.split('.')[0]

def saveLoss(data, settings):
    path = os.path.join('results', 'evaluation')
    file = os.path.join(path, f'{settings}_losses.txt')
    
    str_data = [f'{it}, {batch}, {loss}\n' for it, batch, loss in data]
    
    with open(file, 'w') as f:
        f.write('Iteration, Batch, Loss\n')
        f.writelines(str_data)
        
        
def saveAccuracy(data, type: str, settings):
    path = os.path.join('results', 'evaluation')
    file = os.path.join(path, f'{settings}_{type}_acc.txt')
    
    str_data = [f'{i}, {b[0]}, {b[1]}, {b[2]}, {b[3]}\n' for i, b in enumerate(data)]

    with open(file, 'w') as f:
        f.write('Iteration, BLEU1, BLEU2, BLEU3, BLEU4\n')
        f.writelines(str_data)


if __name__ == '__main__':
    # data = [(1, 2, 3, 4), (2, 3, 4, 5), (3, 4, 5, 6)]
    # saveAccuracy(data, 'something', 'test')
    
    saveSettings()
    pass