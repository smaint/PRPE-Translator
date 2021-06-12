# PRPE-Translator

## Dependencies
```
pip install requirements.txt
```

## Datasets
 - The data we cleaned is found in `data/cleaned_source`. `test.es.txt` and `test.qz.txt` were created by random shuffling of all of the parallel lines. The source data from Annete Rios can be found [here](https://github.com/a-rios/squoia).
 - The Religious, News, and General Indonesian-English datasets from [Benchmarking Multidomain English-Indonesian Machine Translation](https://www.aclweb.org/anthology/2020.bucc-1.6.pdf) can be found at their [repository here](https://github.com/gunnxx/indonesian-mt-data).
 - The Religious and Magazine data from [Neural machine translation with a polysynthetic low resource language ](https://link.springer.com/article/10.1007/s10590-020-09255-9) can be found [here](https://github.com/johneortega/mt_quechua_spanish).

## Running the Code
Our entire pipeline can be run with:
```
python pipeline.py
```
The pipeline can take in several flags:
 - `--segment_type` can be `none`, `bpe`,`unigram`, `prpe`, `prpe_bpe`, `prpe_multi`.
 - `--prpe_multi_runs` is used to set the number of iterations to run the `prpe_multi`(Multi-PRPE). Default value is 5.
 - `--model_type` can be `rnn`(aka LSTM) or `transformer`. Defaults to LSTM.
 - `--in_lang` specifies the input language to be translated. We used `qz` for Quechua and `id` for Indonesian. Defaults to Quechua.
 - `--out_lang` specifies the output language to be translated to. We used `es` for Spanish and `en` for English. Defaults to English.
 - `--domain` specifies the name of dataset to be used, which should be located in `data/` under the same name. Defaults to religious.
 - `--train_steps` specifies how many steps the model should be trained. Default value is 100,000.
 - `--save_steps` specifies how often the trained model is saved. Default is every 10,000 steps.
 - `--validate_steps` specifies how often the model should be evaluated against the validation set. Default is every 2000 steps.
 - `--batch_size` is the batch size for training. Default is 64.
 - `--filter_too_long` specifies the max token length of a line in the training set. Any line that passes this value is filtered out. Default is no filtering.
The pipeline will automatically test the model after training is finished and output a BLEU and CHRF score.

## PRPE
 - From [Semi-automatic Quasi-morphological Word Segmentation for Neural Machine Translation](https://link.springer.com/chapter/10.1007/978-3-319-97571-9_23)
 - The base code for PRPE was taken from https://github.com/zuters/prpe.
