import csv

datasetName='book-crossing'
threshold=6

def scoreToClick(scoreStr):
    score=int(scoreStr)
    if score >= threshold :
        return '1'
    else:
        return '0'

def convertRatingToCTR():
    dataList=[]
    with open(datasetName+'/'+datasetName+'.inter','r') as f:
        reader=csv.reader(f,delimiter='\t')
        for index,line in enumerate(reader):
            if index != 0:
                line[2]=scoreToClick(line[2])
            # dataList.append([line[0]+'\t'+line[1]+'\t'+line[2]+'\t'+line[3]])
            dataList.append([line[0] + '\t' + line[1] + '\t' + line[2]])
    with open(datasetName+'-CTR/'+datasetName+'-CTR.inter','w',newline='') as f:
        writer=csv.writer(f)
        writer.writerows(dataList)

if __name__ == '__main__':
    convertRatingToCTR()

