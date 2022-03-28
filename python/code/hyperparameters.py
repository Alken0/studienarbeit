from tensorboard.plugins.hparams import api as hp
from itertools import product
import constants

HP_BATCH_SIZE = hp.HParam("BatchSize", hp.Discrete([16,32]))

HP_EMBEDDING_SIZE = hp.HParam("Embedding Size", hp.Discrete([50]))

HP_DIS_DROPOUT = hp.HParam("DIS: dropout", hp.Discrete([0.3, 0.4]))
HP_DIS_LR = hp.HParam("DIS: learning_rate", hp.Discrete([2e-3,2e-4, 3e-4]))
HP_DIS_SMOOTH = hp.HParam("DIS: smoothness", hp.Discrete([0.0]))

HP_GEN_LR = hp.HParam("GEN: learning_rate", hp.Discrete([2e-3,2e-4, 3e-4]))

def iterator():
    hp_product = product(
        HP_BATCH_SIZE.domain.values, 
        HP_EMBEDDING_SIZE.domain.values, 
        HP_DIS_DROPOUT.domain.values, 
        HP_DIS_LR.domain.values, 
        HP_DIS_SMOOTH.domain.values, 
        HP_GEN_LR.domain.values
    )
    return hp_product

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

def to_string():
    from constants import BATCH_SIZE, EMBEDDING_SIZE, LEARNING_RATE_DISCRIMINATOR, LEARNING_RATE_GENERATOR, DROPOUT, SMOOTH
    return f"BS-{BATCH_SIZE}_ES-{EMBEDDING_SIZE}_LRD-{LEARNING_RATE_DISCRIMINATOR}_LRG-{LEARNING_RATE_GENERATOR}_DR-{DROPOUT}_SM-{SMOOTH}"

def to_tf_hp():
    from constants import BATCH_SIZE, EMBEDDING_SIZE, LEARNING_RATE_DISCRIMINATOR, LEARNING_RATE_GENERATOR, DROPOUT, SMOOTH
    return {
        HP_BATCH_SIZE: BATCH_SIZE,
        HP_EMBEDDING_SIZE: EMBEDDING_SIZE,
        HP_DIS_LR: LEARNING_RATE_DISCRIMINATOR,
        HP_GEN_LR: LEARNING_RATE_GENERATOR,
        HP_DIS_DROPOUT: DROPOUT,
        HP_DIS_SMOOTH: SMOOTH
    }
