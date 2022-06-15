from recbole_metarec.MetaUtils import metaQuickStart

modelName, datasetName = 'TaNP', 'ml-100k'
metaQuickStart(modelName, datasetName)

'''
    ModelName & DatasetName Available
    
    FOMeLU: ml-100k, ml-1m, book-crossing
    MAMO: ml-100k, ml-1m, book-crossing
    TaNP: ml-100k, ml-1m, book-crossing
    LWA: ml-100k-CTR, ml-1m-CTR, book-crossing-CTR
    NLBA: ml-100k-CTR, ml-1m-CTR, book-crossing-CTR
    MetaEmb: ml-100k-CTR, ml-1m-CTR, book-crossing-CTR
    MWUF: ml-100k-CTR, ml-1m-CTR, book-crossing-CTR
'''