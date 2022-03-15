from tensorboard.plugins.hparams import api as hp
from itertools import product
import constants

HP_BATCH_SIZE = hp.HParam("BatchSize", hp.Discrete([64]))

HP_EMBEDDING_SIZE = hp.HParam("Embedding Size", hp.Discrete([50]))

HP_DIS_DROPOUT = hp.HParam("DIS: dropout", hp.Discrete([0.4]))
HP_DIS_LR = hp.HParam("DIS: learning_rate", hp.Discrete([2e-4, 3e-4]))
HP_DIS_SMOOTH = hp.HParam("DIS: smoothness", hp.Discrete([0, 0.1, 0.2]))

HP_GEN_LR = hp.HParam("GEN: learning_rate", hp.Discrete([2e-4, 3e-4]))

def iterator():
    return product(
        HP_BATCH_SIZE.domain.values, 
        HP_EMBEDDING_SIZE.domain.values, 
        HP_DIS_DROPOUT.domain.values, 
        HP_DIS_LR.domain.values, 
        HP_DIS_SMOOTH.domain.values, 
        HP_GEN_LR.domain.values
    )

def update(element):
    constants.BATCH_SIZE = element[0]
    constants.EMBEDDING_SIZE = element[1]
    constants.DROPOUT = element[2]
    constants.LEARNING_RATE_DISCRIMINATOR = element[3]
    constants.SMOOTH = element[4]
    constants.LEARNING_RATE_GENERATOR = element[5]

    print(f"""Hyperparameters:
    batchsize: {constants.BATCH_SIZE}
    embedding size: {constants.EMBEDDING_SIZE}
    dropout: {constants.DROPOUT}
    lr_dis: {constants.LEARNING_RATE_DISCRIMINATOR}
    smooth: {constants.SMOOTH}
    lr_gen: {constants.LEARNING_RATE_GENERATOR}
    """)
