
import argparse
import logging
import pickle
import re

import numpy as np
import torch

import time

from transformers import (
    AdamW,
    AutoModelWithLMHead,
    AutoTokenizer,
    PreTrainedModel,
)


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

import yaml

import testers
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def plot_distribution2(args, model):
    font = {'size': 5}

    plt.rc('font', **font)

    fig = plt.figure(dpi=300)
    plt.title(args.image_name)
    plt.axis('off')
    
    block_row_division = args.block_row_division
    block_row_width = args.block_row_width

    i = 1
    for name, weight in model.named_parameters():
        if (len(weight.shape)>=2):
            conv = np.abs(weight.cpu().detach().numpy())
            conv = weight.reshape(weight.shape[0], -1)
            print('weight.shape', conv.shape)

            if block_row_width != 0:
                if conv.shape[1]%block_row_width != 0 :
                    print("the layer size is not divisible by block_row_width:",conv.shape[1], block_row_width)
                    # raise SyntaxError("block_size error")
                block_row_division = int(conv.shape[1]/block_row_width)
            else:
                if conv.shape[1]%block_row_division != 0 :
                    print("the layer size is not divisible by block_row_division",conv.shape[1], block_row_division)
                    # raise SyntaxError("block_size error")
            convfrag = torch.chunk(conv, block_row_division, dim=1)

            mat = None
            for k in range(len(convfrag)):
                if mat is None:
                    mat = convfrag[k]                       # if block_row_division = 8, convfrag[j].shape=[64,4]. -libn
                else:
                    mat = torch.cat((mat, convfrag[k]), 0)  # mat.shape=[2400*block_row_division, 800/block_row_division]=[19200, 100] when conv.shape=[2400,800]. -libn

            column_norm = torch.norm(mat, dim=1)
            print('length of counted parameters:', column_norm.shape, '\tparameters.max: %.3f \tparameters.min:%.3f' %(column_norm.max().item(),column_norm.min().item()))
            if len(conv.shape) == 2:
                # xtict_values = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
                xtict_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
                xx = []
                yy = []
                for j, ticks in enumerate(xtict_values):
                    if j == 0:
                        xx.append("<" + str(ticks))
                        yy.append(sum(column_norm < ticks))
                    if j != 0 and j != (len(xtict_values)):
                        xx.append(str(xtict_values[j - 1]) + "~" + str(ticks))
                        # yy.append(len(np.where(np.logical_and(column_norm >= xtict_values[j - 1], column_norm < ticks))[0]))
                        yy.append(len(column_norm[(column_norm >= xtict_values[j - 1])&(column_norm < ticks)]))
                    if j == (len(xtict_values) - 1):
                        xx.append(">=" + str(ticks))
                        yy.append(sum(column_norm >= ticks))
                ax = fig.add_subplot(5, 8, i)
                ax.bar(xx, yy, align='center', color="crimson")  # A bar chart
                # ax.set_title(name)
                ax.set_title(i)
                plt.setp(ax, xticks=xx)
                plt.xticks(rotation=90)
                i += 1
            if i > 40:
                print("!!!!!!!!!!!!!!! Error: Only first 40 layers are displayed !!!!!!!!!!!!!!!!")
                break
    # plt.show()
    plt.savefig(args.image_name+'.jpg')
    print("Param distribution image saved!")

def main():

    parser = argparse.ArgumentParser()

    # parser.add_argument('--sparsity_type', type=str, default='column',
    #                 help ="define sparsity_type: [irregular,column,filter]")
    parser.add_argument('--block_row_division', type=int, default=8,
                    help='the number of division for each row for block-wise pruning')
    parser.add_argument('--block_row_width', type=int, default=0,
                    help='the width of the pruned block for each row')

    parser.add_argument('--cross_f', type=int, default=8,
                    help='the crossbar filter number, set to 1 to disable') 
    # parser.add_argument('--load_model', type=str, default='./model_reweighted/temp_progressive.pt',
    #                 help ="load model for test")
    parser.add_argument('--image_name', type=str, default='results',
                    help ="image_name of parameters distribution")

    parser.add_argument(
        "--output_dir",
        type=str,
        default = 'output',
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    args = parser.parse_args()

    device = torch.device("cuda")

    # load trained model:
    # Load a trained model and vocabulary that you have fine-tuned
    model = AutoModelWithLMHead.from_pretrained(args.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
    model.to(device)
    # print(model)

    testers.test_irregular_sparsity(model)

    plot_distribution2(args, model)

if __name__ == "__main__":
    main()


# python check_model.py --output_dir output/checkpoint-500 --image_name param_dis
