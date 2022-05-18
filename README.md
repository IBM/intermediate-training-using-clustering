# Intermediate Training Using Clustering

Code to reproduce the BERT intermediate training experiments from [Shnarch et al. (2022)](#reference). 

Using this repository you can: 

(1) Download the datasets used in the paper;

(2) Run intermediate training that relies on pseudo-labels from the results of the [sIB](https://github.com/IBM/sib) clustering algorithm;

(3) Fine-tune a BERT classifier starting from the default pretrained model (bert-base-uncased) and from the model after intermediate training;

(4) Compare the the BERT classification performance with and without the intermediate training stage.


**Table of contents**

[Installation](#installation)

[Running an experiment](#running-an-experiment)

[Plotting the results](#plotting-the-results)

[Reference](#reference)

[License](#license)

## Installation
The framework requires Python 3.8
1. Clone the repository locally: 
   `git clone https://github.com/IBM/intermediate-training-using-clustering`
2. Go to the cloned directory 
  `cd intermediate-training-using-clustering`
4. Install the project dependencies: `pip install -r requirements.txt`

   Windows users may also need to download the latest [Microsoft Visual C++ Redistributable for Visual Studio](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads) in order to support tensorflow
3. Run the python script `python download_and_process_datasets.py`.
This script downloads and processes 8 datasets used in the paper.

                         
## Running an experiment
The experiment script `run_experiment.py` requires 6 arguments: 
- `train_file`: path to the train data (e.g. datasets/isear/train.csv). 
- `eval_file`: path to the evaluation data (e.g. datasets/isear/test.csv). 
- `num_clusters`: number of clusters used to generate the task pseudo labels. Defaults to 50 (as used in the paper) 
- `labeling_budget`: number of examples from the train data used for BERT fine-tuning (in the paper we tested the following budgets: 64, 128, 192, 256, 384, 512, 768, 1024)
- `random_seed`: used for sampling the train data and for model training
- `inter_training_epochs`: number of epochs for the intermediate task. Defaults to 1 (as used in the paper)
- `finetuning_epochs`: number of epochs for fine-tuning BERT over `labeling_budget` examples. Defaults to 10 (as used in the paper)

For example: 

```python run_experiment.py --train_file datasets/yahoo_answers/train.csv --eval_file datasets/yahoo_answers/test.csv --num_clusters 50 --labeling_budget 64 --finetuning_epochs 10 --inter_training_epochs 1 --random_seed 0```

The results of the experimental run (accuracy for BERT with and without the intermediate task over the `eval_file`) are written both to the screen, and to `output/results.csv`. 

Multiple experiments can safely write in parallel to the same `output/results.csv` file - each new result is appended to the file. In addition, for every new result, an aggregation of all the results so far is written to `output/aggregated_results.csv`. This aggregation reflects the mean of all runs for each experimental setting (i.e. with/without intermediate training) for a particular eval_file and labeling budget.


## Plotting the results
In order to show the effect of the intermediate task in different labeling budgets, run `python plot.py`. This script generates plots under `output/plots` for each dataset.

For example:


![Alt text](example_plot.png?raw=true "Output image of plot.py after running 5 seeds over 8 labeling budgets for dbpedia")


## Reference
Eyal Shnarch, Ariel Gera, Alon Halfon, Lena Dankin, Leshem Choshen, Ranit Aharonov and Noam Slonim (2022). 
[Cluster & Tune: Boost Cold Start Performance in Text Classification](https://aclanthology.org/2022.acl-long.526/). ACL 2022

Please cite: 
```
@inproceedings{shnarch-etal-2022-cluster,
    title = "Cluster & Tune: Boost Cold Start Performance in Text Classification",
    author = "Shnarch, Eyal  and
      Gera, Ariel  and
      Halfon, Alon  and
      Dankin, Lena  and
      Choshen, Leshem  and
      Aharonov, Ranit  and
      Slonim, Noam",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.526",
    pages = "7639--7653",
}
```

## License
This work is released under the Apache 2.0 license. The full text of the license can be found in [LICENSE](LICENSE).

