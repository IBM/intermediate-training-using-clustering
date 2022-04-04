# © Copyright IBM Corporation 2022.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy import stats


df = pd.read_csv('./output/results.csv')
df['dataset_name'] = df['eval_file'].apply(lambda x: x.split('/')[-2])
datasets = ['ag_news', 'isear', 'dbpedia', '20_newsgroup', 'yahoo_answers', 'polarity', 'subjectivity', 'sms_spam']

for dataset in datasets:
    for setting in ['base', 'intermediate']:
        setting_df = df.query('setting_name == @setting & dataset_name == @dataset')
        aggregated_df = setting_df.groupby('labeling_budget', as_index=False).mean()
        x = aggregated_df['labeling_budget']
        y = aggregated_df['accuracy']
        plt.plot(x, y, label=setting)
        stderr = [stats.sem(x['accuracy']) for _, x in setting_df.groupby('labeling_budget', as_index=False)]
        plt.fill_between(x, y - stderr, y + stderr, alpha=0.2)

        plt.title(dataset)
        plt.legend()
    os.makedirs("output/plots", exist_ok=True)
    plt.savefig(f"output/plots/{dataset}.png")
    plt.close()