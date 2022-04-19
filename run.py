from recbole.utils import init_logger, init_seed
from recbole.config import Config
from MetaUtils import *
from model.FOMeLU.FOMeLUTrainer import FOMeLUTrainer
from model.FOMeLU.FOMeLU import FOMeLU

modelName='FOMeLU'
datasetName='ml-100k'
trainerName=modelName+'Trainer'
configPath=['model/'+modelName+'/'+modelName+'.yaml']

if __name__ == '__main__':
    config = Config(model=eval(modelName), dataset=datasetName, config_file_list=configPath)
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering
    dataset = create_meta_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = meta_data_preparation(config, dataset)
    logger.info(train_data)

    # model loading and initialization
    model = eval(modelName)(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = eval(trainerName)(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    # model evaluation
    test_result = trainer.evaluate(test_data)

    logger.info('best valid result: {}'.format(best_valid_result))
    logger.info('test result: {}'.format(test_result))

