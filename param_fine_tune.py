import importlib,csv
from recbole.utils import init_logger, init_seed
from recbole.config import Config
from MetaUtils import *

def runSingleModel(modelName,datasetName,param_dict,logger=False):
    trainerClass = importlib.import_module('model.' + modelName + '.' + modelName + 'Trainer').__getattribute__(
        modelName + 'Trainer')
    modelClass = importlib.import_module('model.' + modelName + '.' + modelName).__getattribute__(modelName)
    config_file_listPath = ['model/' + modelName + '/' + modelName + '.yaml']
    if datasetName == 'book-crossing' or datasetName == 'book-crossing-CTR':
        config_file_listPath = ['model/' + modelName + '/' + modelName + '-BK.yaml']

    config = Config(model=modelClass, dataset=datasetName, config_file_list=config_file_listPath,config_dict=param_dict)
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    if logger:
        init_logger(config)
        logger = getLogger()
        logger.info(config)

    # dataset filtering
    dataset = create_meta_dataset(config)
    if logger:
        logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = meta_data_preparation(config, dataset)
    if logger:
        logger.info(train_data)

    # model loading and initialization
    model = modelClass(config, train_data.dataset).to(config['device'])
    if logger:
        logger.info(model)

    # trainer loading and initialization
    trainer = trainerClass(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    # model evaluation
    test_result = trainer.evaluate(test_data)
    if logger:
        logger.info('best valid result: {}'.format(best_valid_result))
        logger.info('test result: {}'.format(test_result))

    return test_result

typeDict={
    'FOMeLU':'rating',
    'MAMO':'rating',
    'TaNP':'rating',
    'LWA':'click',
    'NLBA':'click',
    'MetaEmb':'click',
    'MWUF':'click',
}

FOMeLUTuneDict=OrderedDict({
        'local_lr':[0.000005,0.0005,0.005],
        'lr':[0.00005,0.005,0.05]
    })
MAMOTuneDict=OrderedDict({
        'alpha':[0.1,0.2,0.5],
        'beta':[0.05,0.1,0.2]
    })
TaNPTuneDict=OrderedDict({
        'lr':[0.0001,0.001,0.005,0.01,0.02,0.05,0.1,0.2]
    })
LWATuneDict=OrderedDict({
        'lr':[0.0001,0.001,0.005,0.01,0.02,0.05,0.1,0.2]
    })
NLBATuneDict=OrderedDict({
        'lr':[0.0001,0.001,0.005,0.01,0.02,0.05,0.1,0.2]
    })
MetaEmbTuneDict=OrderedDict({
        'local_lr':[0.0001,0.001,0.01],
        'lr':[0.0001,0.001,0.01]
    })
MWUFTuneDict=OrderedDict({
        'local_lr':[0.0001,0.001,0.01],
        'lr':[0.0001,0.001,0.01]
    })

rankingMetrics=['precision','recall','hit','ndcg','mrr']

def dfs(total,now,values):
    if len(values) == 1:
        total+=[now+[item] for item in values[0]]
        return

    for item in values[0]:
        dfs(total,now+[item],values[1:])

def getDetail(parameter_dict):
    detail=''
    for k,v in parameter_dict.items():
        detail+=k+'('+str(v)+');'
    return detail

def finetuneSingleModel(modelName,datasetName,tuneDict,logger=False):
    oriDTN=datasetName
    valueMetrics = ['mae']
    if typeDict[modelName] == 'click':
        datasetName=datasetName+'-CTR'
        valueMetrics = ['auc']

    values=list(tuneDict.values())
    keys=list(tuneDict.keys())

    paramList=[]
    dfs(paramList,[],values)

    board=[['ModelName','Dataset','Detail']+valueMetrics+rankingMetrics]
    # board = [['ModelName', 'Dataset', 'Detail']+ rankingMetrics]

    print(modelName,'start with',len(paramList),'groups.')
    for index,param in enumerate(paramList):
        parameter_dict=dict(zip(keys,param))
        detail=getDetail(parameter_dict)

        parameter_dict['epochs']=50
        parameter_dict['metrics']=rankingMetrics
        parameter_dict['valid_metric']='mrr@5'
        rankingResult=runSingleModel(modelName,datasetName,parameter_dict,logger)

        parameter_dict['metrics']=valueMetrics
        parameter_dict['valid_metric'] = 'mae'
        valueResult=runSingleModel(modelName,datasetName,parameter_dict,logger)
        valueResult.update(rankingResult)

        board.append([modelName]+[datasetName]+[detail]+[v for k,v in valueResult.items()])
        # board.append([modelName] + [datasetName] + [detail] + [v for k, v in rankingResult.items()])
        print('Finish group',index)

    path='performance/'+oriDTN+'/'+modelName+'.csv'
    with open(path,'w',newline='') as f:
        csvwriter=csv.writer(f)
        csvwriter.writerows(board)
    print(modelName,'finish.')

def fintuneAllModels(modelList,datasetName,logger=False):
    for modelName in modelList:
        tuneDict = eval(modelName + 'TuneDict')
        finetuneSingleModel(modelName, datasetName, tuneDict,logger)

if __name__ == '__main__':
    # modelList = ['FOMeLU','MAMO','TaNP','LWA','NLBA','MetaEmb','MWUF']
    modelList = ['MWUF']
    datasetName='book-crossing'
    fintuneAllModels(modelList,datasetName,logger=True)

