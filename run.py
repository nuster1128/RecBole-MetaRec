from recbole.utils import init_logger, init_seed
from recbole.config import Config
from MetaUtils import *
'''from model.MeLU.MeLUTrainer import MeLUTrainer
from model.MeLU.MeLU import MeLU'''
from model.NLBA.NLBA import NLBA
from model.NLBA.NLBATrainer import NLBATrainer

if __name__ == '__main__':
    '''config = Config(model=MeLU, dataset='ml-100k-local',config_file_list=['model/MeLU/MeLU.yaml'])
    init_seed(config['seed'], config['reproducibility'])'''
    config = Config(model=NLBA, dataset='ml-100k-local-CTR', config_file_list=['model/NLBA/NLBA.yaml'])
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
    '''model = MeLU(config, train_data.dataset).to(config['device'])'''
    model = NLBA(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    '''trainer = MeLUTrainer(config, model)'''
    trainer = NLBATrainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    # model evaluation
    test_result = trainer.evaluate(test_data)

    logger.info('best valid result: {}'.format(best_valid_result))
    logger.info('test result: {}'.format(test_result))