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
## ml-100k
kFOMeLUTuneDict=OrderedDict({
        'embedding_size': [16],
        'train_batch_size': [256],
        'lr': [0.01],
        'mlp_hidden_size': [[8,8],[16,16],[32,32],[64,64],[128,128],[256,256]]
    })
kTaNPTuneDict=OrderedDict({
        'embedding': [256],
        'train_batch_size': [128],
        'lr': [0.01],
        'lambda': [0.05,0.1,0.2,0.5,0.8,1.0]
    })
kLWATuneDict=OrderedDict({
        'embedding': [8],
        'train_batch_size': [8],
        'lr': [0.01],
        'embeddingHiddenDim': [8,16,32,64,128,256]
    })
kNLBATuneDict=OrderedDict({
        'embedding': [16],
        'train_batch_size': [8],
        'lr': [0.01],
        'recHiddenDim': [8,16,32,64,128,256]
    })
kMetaEmbTuneDict=OrderedDict({
        'embedding': [128],
        'train_batch_size': [8],
        'lr': [0.01],
        'alpha': [0.05,0.1,0.2,0.5,0.8,1.0]
    })
kMWUFTuneDict=OrderedDict({
        'embedding': [256],
        'train_batch_size': [8],
        'warmLossLr': [0.1],
        'indexEmbDim': [8,16,32,64,128,256]
    })

## ml-1m
mFOMeLUTuneDict=OrderedDict({
        'embedding_size': [8],
        'train_batch_size': [8],
        'lr': [0.01],
        'mlp_hidden_size': [[8,8],[16,16],[32,32],[64,64],[128,128],[256,256]]
    })
mTaNPTuneDict=OrderedDict({
        'embedding': [16],
        'train_batch_size': [8],
        'lr': [0.01],
        'lambda': [0.05,0.1,0.2,0.5,0.8,1.0]
    })
mLWATuneDict=OrderedDict({
        'embedding': [8],
        'train_batch_size': [8],
        'lr': [0.2],
        'embeddingHiddenDim': [8,16,32,64,128,256]
    })
mNLBATuneDict=OrderedDict({
        'embedding': [8],
        'train_batch_size': [8],
        'lr': [0.01],
        'recHiddenDim': [8,16,32,64,128,256]
    })
mMetaEmbTuneDict=OrderedDict({
        'embedding': [256],
        'train_batch_size': [8],
        'lr': [0.5],
        'alpha': [0.05,0.1,0.2,0.5,0.8,1.0]
    })
mMWUFTuneDict=OrderedDict({
        'embedding': [256],
        'train_batch_size': [64],
        'warmLossLr': [0.05],
        'indexEmbDim': [8,16,32,64,128,256]
    })

## bookcrossing
gFOMeLUTuneDict=OrderedDict({
        'embedding_size': [8],
        'train_batch_size': [8],
        'lr': [0.01],
        'mlp_hidden_size': [[8,8],[16,16],[32,32],[64,64],[128,128],[256,256]]
    })
gTaNPTuneDict=OrderedDict({
        'embedding': [8],
        'train_batch_size': [8],
        'lr': [0.01],
        'lambda': [0.05,0.1,0.2,0.5,0.8,1.0]
    })
gLWATuneDict=OrderedDict({
        'embedding': [64],
        'train_batch_size': [8],
        'lr': [0.01],
        'embeddingHiddenDim': [8,16,32,64,128,256]
    })
gNLBATuneDict=OrderedDict({
        'embedding': [16],
        'train_batch_size': [128],
        'lr': [0.01],
        'recHiddenDim': [8,16,32,64,128,256]
    })
gMetaEmbTuneDict=OrderedDict({
        'embedding': [32],
        'train_batch_size': [8],
        'lr': [0.01],
        'alpha': [0.05,0.1,0.2,0.5,0.8,1.0]
    })
gMWUFTuneDict=OrderedDict({
        'embedding': [16],
        'train_batch_size': [8],
        'warmLossLr': [0.01],
        'indexEmbDim': [8,16,32,64,128,256]
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
    if typeDict[modelName] == 'click':
        datasetName=datasetName+'-CTR'

    values=list(tuneDict.values())
    keys=list(tuneDict.keys())

    paramList=[]
    dfs(paramList,[],values)

    board = [['ModelName', 'Dataset', 'Detail']+ rankingMetrics]

    print(modelName,'start with',len(paramList),'groups.')
    for index,param in enumerate(paramList):
        parameter_dict=dict(zip(keys,param))
        detail=getDetail(parameter_dict)

        parameter_dict['epochs']=10
        parameter_dict['metrics']=rankingMetrics
        parameter_dict['valid_metric']='mrr@5'
        rankingResult=runSingleModel(modelName,datasetName,parameter_dict,logger)

        board.append([modelName] + [datasetName] + [detail] + [v for k, v in rankingResult.items()])
        print('Finish group',index)

    path='performance/'+oriDTN+'/'+modelName+'.csv'
    with open(path,'w',newline='') as f:
        csvwriter=csv.writer(f)
        csvwriter.writerows(board)
    print(modelName,'finish.')

def fintuneAllModels(modelList,datasetName,logger=False):
    for modelName in modelList:
        tuneDict = eval(datasetName[-1]+modelName + 'TuneDict')
        finetuneSingleModel(modelName, datasetName, tuneDict,logger)

if __name__ == '__main__':
    # modelList = ['FOMeLU','MAMO','TaNP','LWA','NLBA','MetaEmb','MWUF']
    modelList = ['FOMeLU','TaNP','LWA','NLBA','MetaEmb','MWUF']
    datasetName='ml-100k'
    fintuneAllModels(modelList,datasetName,logger=False)
    modelList = ['FOMeLU','TaNP','LWA','NLBA','MetaEmb','MWUF']
    datasetName='ml-1m'
    fintuneAllModels(modelList,datasetName,logger=False)
    modelList = ['FOMeLU','TaNP','LWA','NLBA','MetaEmb','MWUF']
    datasetName='book-crossing'
    fintuneAllModels(modelList,datasetName,logger=False)
