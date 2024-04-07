import time
import argparse
import os

def get_dataset_name():
    fns = [fn.split('.')[0] for fn in os.listdir('./datasets')]
    return fns

model_names = ['EODL']
file_names = get_dataset_name()

arg_parser = argparse.ArgumentParser(description='EODL main script')

# experiment related
exp_group = arg_parser.add_argument_group('exp', 'experiment setting')

exp_group.add_argument('-r','--root', default='Experiment',
                       type=str, help='root path to the experiment result'
                       '(default: ./Experiment/)')

exp_group.add_argument('-f','--filename', default='result.json',
                       type=str, help='filename of the experiment results file'
                       '(default: result.json)')

exp_group.add_argument('--bs', default=1, type=int, 
                         help='batch size')

exp_group.add_argument('--lr', default=0.01, type=float, 
                         help='learning rate')


# dataset related
data_group = arg_parser.add_argument_group('data', 'dataset setting')

data_group.add_argument('-d','--dataset',default='hyperplane',type=str,
                       help='datasets: ' +
                        ' | '.join(file_names) +
                        ' (default: elec)')
# network related
net_group = arg_parser.add_argument_group('network', 'network setting')

net_group.add_argument('-m','--model',default='EODL',type=str,
                       help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: EODL)')

net_group.add_argument('--ln',default=20,type=int,
                       help='layers number (default:20)')

net_group.add_argument('--nHn',default=100, type=int,
                       help='neurons number of hidden layers (default:100)')

net_group.add_argument('--beta', default=0.80, type=float,
                       help='declay factor of prediction weight vector ')

net_group.add_argument('--theta', default=0.01, type=float,
                       help='concept drift detection threshold')

net_group.add_argument('--smooth', default=0.2, type=float,
                       help='concept drift detection threshold')

net_group.add_argument('-p', default=0.99, type=float,
                       help='weighted coefficient of exponential moving average')

net_group.add_argument('--del-da', action='store_true', help='EODL without depth adpation strategy')

net_group.add_argument('--del-pa', action='store_true', help='EODL without parameter adaption strategy')

net_group.add_argument('--del-all', action='store_true', help='EODL without depth adpation strategy and parameter adaption strategy')



