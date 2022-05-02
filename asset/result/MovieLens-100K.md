## Experiment Settings

**Dataset:** MovieLens-100K

**Metircs:** Precision@5, Recall@5, HR@5, nDCG@5,MRR@5

**Task defination:** We format the dataset into tasks and each user can be represented as a task. Therefore, we set the task proportion of training: validation: test as 8:1:1. For each task, we randomly select **10 interactions as query set**, and **the others as support set**.

**Data type and filtering:** We use MovieLens-100K for the rating and click experiment respectively. For rating settings, we use the original rating scores. For click settings, as many papers do, we consider rating scores equal or above 4 as positive labels, and others are negative. Moreover, we set the user interaction number interval as [13,100] as many papers do.

The common configurations are listed as follows.

```yaml
# Dataset config
USER_ID_FIELD: user_id
ITEM_ID_FIELD: item_id

load_col:
    inter: [user_id, item_id, rating]
    item: [item_id,movie_title,release_year,class]
    user: [user_id,age,gender,occupation,zip_code]
user_inter_num_interval: [13,100]

# Training and evaluation config
epochs: 10
train_batch_size: 32
valid_metric: mrr@5

# Evaluate config
eval_args:
    group_by: task
    order: RO
    split: {'RS': [0.8,0.1,0.1]}
    mode : labeled

# Meta learning config
meta_args:
    support_num: none
    query_num: 10

# Metrics
metrics: ['precision','recall','hit','ndcg','mrr']
metric_decimal_place: 4
topk: 5
```

## Evaluation Results

<table>
  <caption>MovieLens-100K Performance</caption>
  <tr>
  	<th>MovieLens-100K</th>
    <th>Type</th>
    <th>Data</th>
    <th>Precision@5</th>
    <th>Recll@5</th>
    <th>HR@5</th>
    <th>nDCG@5</th>
    <th>MRR@5</th>
  </tr>
  <tr>
  	<td>MeLU</td>
    <td rowspan="2">Learn to predict</td>
    <td>Rating</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
  	<td>MAMO</td>
    <td>Rating</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
  	<td>TaNP</td>
    <td rowspan="3">Learn to parameterize</td>
    <td>Rating</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
  	<td>LWA</td>
    <td>Click</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
  	<td>NLBA</td>
    <td>Click</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
  	<td>MetaEmb</td>
    <td rowspan="2">Learn to embedding</td>
    <td>Click</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
  	<td>MWUF</td>
    <td>Click</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</table>

## Hyper Parameter Tuning

<table>
  <tr>
  	<th>Model</th>
    <th>Best Hyper Parameter</th>
    <th>Tuning Range</th>
  </tr>
  <tr>
    <td><b>FOMeLU</b></td>
    <td>embedding_size: [16];<br>
      train_batch_size: [256];<br>
      lr: [0.01];<br>
      mlp_hidden_size: [[64,64]]</td>
    <td>embedding_size: [8,16,32,64,128,256];<br>
      train_batch_size: [8,16,32,64,128,256];<br>
      lr: [0.0001,0.001,0.01,0.05,0.1,0.2,0.5,1.0];<br>
      mlp_hidden_size: [[8,8],[16,16],[32,32],[64,64],[128,128],[256,256]]</td>
  </tr>
  <tr>
  	<td><b>MAMO</b></td>
    <td>embedding: [8];<br>
      train_batch_size: [8];<br>
      lambda (lr): [0.01];<br>
      beta: [0.05]</td>
    <td>embedding: [8,16,32,64,128,256];<br>
      train_batch_size: [8,16,32,64,128,256];<br>
      lambda (lr): [0.0001,0.001,0.01,0.05,0.1,0.2,0.5,1.0];<br>
      beta: [0.05,0.1,0.2,0.5,0.8,1.0]</td>
  </tr>
  <tr>
    <td><b>TaNP</b></td>
    <td>embedding: [256];<br>
      train_batch_size: [128];<br>
      lr: [0.01];<br>
      lambda: [0.05]</td>
    <td>embedding: [8,16,32,64,128,256];<br>
      train_batch_size: [8,16,32,64,128,256];<br>
      lr: [0.0001,0.001,0.01,0.05,0.1,0.2,0.5,1.0];<br>
      lambda: [0.05,0.1,0.2,0.5,0.8,1.0]</td>
  </tr>
  <tr>
    <td><b>LWA</b></td>
    <td>embedding_size: [8];<br>
      train_batch_size: [8];<br>
      lr: [0.01];<br>
      embeddingHiddenDim: [256]</td>
    <td>embedding_size: [8,16,32,64,128,256];<br>
      train_batch_size: [8,16,32,64,128,256];<br>
      lr: [0.0001,0.001,0.01,0.05,0.1,0.2,0.5,1.0];<br>
      embeddingHiddenDim: [8,16,32,64,128,256]</td>
  </tr>
  <tr>
    <td><b>NLBA</b></td>
    <td>embedding_size: [16];<br>
      train_batch_size: [8];<br>
      lr: [0.01];<br>
      recHiddenDim: [32]</td>
    <td>embedding_size: [8,16,32,64,128,256];<br>
      train_batch_size: [8,16,32,64,128,256];<br>
      lr: [0.0001,0.001,0.01,0.05,0.1,0.2,0.5,1.0];<br>
      recHiddenDim: [8,16,32,64,128,256]</td>
  </tr>
  <tr>
    <td><b>MetaEmb</b></td>
    <td>embedding_size: [128];<br>
      train_batch_size: [8];<br>
      lr: [0.01];<br>
      alpha: [0.2]</td>
    <td>embedding_size: [8,16,32,64,128,256];<br>
      train_batch_size: [8,16,32,64,128,256];<br>
      lr: [0.0001,0.001,0.01,0.05,0.1,0.2,0.5,1.0];<br>
      alpha: [0.05,0.1,0.2,0.5,0.8,1.0]</td>
  </tr>
  <tr>
    <td><b>MWUF</b></td>
    <td>embedding_size: [256];<br>
      train_batch_size: [8];<br>
      warmLossLr: [0.1];<br>
      indexEmbDim: [64]</td>
    <td>embedding_size: [8,16,32,64,128,256];<br>
      train_batch_size: [8,16,32,64,128,256];<br>
      warmLossLr: [0.0001,0.001,0.01,0.05,0.1,0.2,0.5,1.0];<br>
      indexEmbDim: [8,16,32,64,128,256]</td>
  </tr>
</table>
