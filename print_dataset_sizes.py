# © Copyright IBM Corporation 2022.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0
import os
import pandas as pd

if __name__ == '__main__':
    DATASETS_DIR = "./datasets"
    for sub_dir in os.listdir(DATASETS_DIR):
        if sub_dir != 'raw':
            print(sub_dir)
            print("\t", len(pd.read_csv(os.path.join(DATASETS_DIR, sub_dir, "train.csv"))),
                  len(pd.read_csv(os.path.join(DATASETS_DIR, sub_dir, "dev.csv"))),
                  len(pd.read_csv(os.path.join(DATASETS_DIR, sub_dir, "test.csv"))))
