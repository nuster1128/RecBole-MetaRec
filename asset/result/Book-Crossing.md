## Experiment Settings

**Dataset:** Book-Crossing

**Metircs:** Precision@5, Recall@5, HR@5, nDCG@5,MRR@5

**Task defination:** We format the dataset into tasks and each user can be represented as a task. Therefore, we set the task proportion of training: validation: test as 8:1:1. For each task, we randomly select **10 interactions as query set**, and **the others as support set**.

**Data type and filtering:** We use Book-Crossing dataset for the rating and click experiment respectively. For rating settings, we use the original rating scores. For click settings, as many papers do, we consider rating scores equal or above 4 as positive labels, and others are negative. Moreover, we set the user interaction number interval as [13,100] as many papers do.

The common configurations are listed as follows.

```yaml
# Dataset config
USER_ID_FIELD: user_id
ITEM_ID_FIELD: item_id
RATING_FIELD: label
LABEL_FIELD: label
load_col:
    inter: [user_id, item_id, label]
    item: [item_id,book_title,book_author,publication_year,publisher]
    user: [user_id,location,age]
user_inter_num_interval: [13,100]

# Training and evaluation config
epochs: 50
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

In addition, we constrain the embedding size with `64`.

\* Because MAMO has a huge demand for GPU memory, we set the ` epochs` with `2` for fairness. Acoording to the training loss and validation results, it has been converged in all probability.

## Evaluation Results

<table>
  <caption>Book-Crossing Performance</caption>
  <tr>
  	<th>Book-Crossing</th>
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
    <td>0.5058</td>
    <td>0.5058</td>
    <td>0.9961</td>
    <td>0.5029</td>
    <td>0.6967</td>
  </tr>
  <tr>
  	<td>MAMO</td>
    <td>Rating</td>
    <td>0.6413</td>
    <td>0.6413</td>
    <td>1.0000</td>
    <td>0.6621</td>
    <td>0.8343</td>
  </tr>
  <tr>
  	<td>TaNP</td>
    <td rowspan="3">Learn to parameterize</td>
    <td>Rating</td>
    <td>0.6413</td>
    <td>0.6413</td>
    <td>1.0000</td>
    <td>0.6621</td>
    <td>0.8343</td>
  </tr>
  <tr>
  	<td>LWA</td>
    <td>Click</td>
    <td>0.7255</td>
    <td>0.7255</td>
    <td>1.0000</td>
    <td>0.7506</td>
    <td>0.8951</td>
  </tr>
  <tr>
  	<td>NLBA</td>
    <td>Click</td>
    <td>0.7255</td>
    <td>0.7255</td>
    <td>1.0000</td>
    <td>0.7506</td>
    <td>0.8951</td>
  </tr>
  <tr>
  	<td>MetaEmb</td>
    <td rowspan="2">Learn to embedding</td>
    <td>Click</td>
    <td>0.7194</td>
    <td>0.7194</td>
    <td>1.0000</td>
    <td>0.7461</td>
    <td>0.8950</td>
  </tr>
  <tr>
  	<td>MWUF</td>
    <td>Click</td>
    <td>0.7194</td>
    <td>0.7194</td>
    <td>1.0000</td>
    <td>0.7461</td>
    <td>0.8950</td>
  </tr>
</table>


## Hyper Parameter Tuning

| Model       | Best Hyper Parameter | Tuning Range                                             |
| ----------- | -------------------- | -------------------------------------------------------- |
| **FOMeLU**  | All Same Performance | local_lr:[0.000005,0.0005,0.005],lr:[0.00005,0.005,0.05] |
| **MAMO**    | All Same Performance | alpha:[0.1,0.2,0.5], beta:[0.05,0.1,0.2]                 |
| **TaNP**    | All Same Performance | lr:[0.0001,0.001,0.005,0.01,0.02,0.05,0.1,0.2]           |
| **LWA**     | lr=0.001             | lr:[0.0001,0.001,0.005,0.01,0.02,0.05,0.1,0.2]           |
| **NLBA**    | All Same Performance | lr:[0.0001,0.001,0.005,0.01,0.02,0.05,0.1,0.2]           |
| **MetaEmb** | All Same Performance | local_lr:[0.0001,0.001,0.01], lr:[0.0001,0.001,0.01]     |
| **MWUF**    | All Same Performance | local_lr:[0.0001,0.001,0.01], lr:[0.0001,0.001,0.01]     |
