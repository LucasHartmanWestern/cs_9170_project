from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
warnings.filterwarnings("ignore")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np

from options import Options
from lib.data import MinMaxScaler
from lib.timegan import TimeGAN

from lib.discriminative_metrics import discriminative_score_metrics
from lib.predictive_metrics import predictive_score_metrics
from lib.visualization_metrics import visualization


def load_data(seq_len=1000):
    ori_data = np.loadtxt('formatted_data.csv', delimiter=',', skiprows=1)
    print(ori_data.shape)

    # Normalize the data
    ori_data = MinMaxScaler(ori_data)

    temp_data = []    
    # Cut data by sequence length
    for i in range(0, len(ori_data) - seq_len):
        _x = ori_data[i:i + seq_len]
        temp_data.append(_x)
            
    # Mix the datasets (to make it similar to i.i.d)
    idx = np.random.permutation(len(temp_data))    
    data = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])
        
    return data


def evaluate(opt, ori_data, generated_data):
    metric_iteration = opt.metric_iteration

    discriminative_score = list()
    for _ in range(metric_iteration):
        temp_disc = discriminative_score_metrics(ori_data, generated_data)
        discriminative_score.append(temp_disc)

    predictive_score = list()
    for tt in range(metric_iteration):
        temp_pred = predictive_score_metrics(ori_data, generated_data)
        predictive_score.append(temp_pred)   

    file_name = os.path.join(opt.outf, opt.name, 'eval_metrics.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write(f'Discriminative score: {str(np.round(np.mean(discriminative_score), 4))}\n')
        opt_file.write(f'Predictive score: {str(np.round(np.mean(predictive_score), 4))}\n')

    print(f'Discriminative score: {str(np.round(np.mean(discriminative_score), 4))}')
    print(f'Predictive score: {str(np.round(np.mean(predictive_score), 4))}')


def visualize(opt, ori_data, generated_data):
    save_path = os.path.join(opt.outf, opt.name, "figures")
    for metric in ['pca', 'tsne']:
        visualization(ori_data, generated_data, metric, save_path)


def train():
    """ Training
    """

    # ARGUMENTS
    opt = Options().parse()
    print("Arguments parsed.")

    # LOAD DATA
    ori_data = load_data()
    print("Dataset loaded.")

    # LOAD MODEL
    model = TimeGAN(opt, ori_data)
    print("Model compiled.")

    # TRAIN MODEL
    model.train()
    print("Training complete.")

    # EVALUATE MODEL
    evaluate(opt, ori_data, model.generated_data)
    print("Evaluation complete.")

    # VISUALIZE
    visualize(opt, ori_data, model.generated_data)
    print("Figures created.")


if __name__ == '__main__':
    train()
