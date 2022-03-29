## RecBole-MetaRec

The document can be found in https://recbole-metarec-doc.readthedocs.io/en/latest/.

## Introduction

RecBole-MetaRec is an extended module for RecBole, which aims to help researches to compare and develop their own models in meta learning recommendation field.

This module is totally developed based on RecBole by adding extened classes and functions, without modifying any codes of RecBole core.

The contributions are briefly listed as follows:

- We extend `MetaDataset` from `Dataset` to split dataset by 'task'.
- We extend `MetaDataLoader` from `AbstractDataLoader` to transform dataset into task form.
- We extend `MetaRecommender` from `AbstractRecommender` to provide a base recommender for implementing meta learning model.
- We extend `MetaTrainer` from  `Trainer` to provide a base trainer for implementing meta learning training process.
- We extend `MetaCollector` `Collector` to collect data for evaluation in meta learning circumstance.
- We implement `MetaUtils` with some useful toolkits for meta learning.

Therefore, researches can:

- Conveniently develop their own meta learning recommendation models.
- Conveniently learn and compare meta learning recommendation models that we have implemented.
- Enjoy advantages and features of RecBole.

Before start, it is strongly recommended to realize how RecBole works, and the homepage of RecBole is https://recbole.io.

Details can be found in the package document.
