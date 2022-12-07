# imports
from typing import Literal
import numpy as np

def evaluate(
    encoder,
    decoder,
    type: Literal['train', 'dev', 'test'],
):
    a1, a2, a3, a4 = sorted(np.random.rand(4), reverse=True)
    return (a1, a2, a3, a4)


if __name__ == '__main__':
    evaluate()