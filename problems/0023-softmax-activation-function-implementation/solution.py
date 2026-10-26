import math
import numpy as np

def softmax(scores: list[float]) -> list[float]:
    # Your code here
    softmaxed_scores = []
    for i in range(len(scores)):
        softmax = np.exp(scores[i]-np.max(scores))/np.sum(np.exp(scores-np.max(scores)))
        softmaxed_scores.append(softmax)
    return softmaxed_scores    