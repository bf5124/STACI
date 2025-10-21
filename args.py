import os
import argparse
import torch
import random
from pprint import pprint


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class Parser(argparse.ArgumentParser):
    def __init__(self):
        super(Parser, self).__init__(description='STACI')
       
        # experiment
        self.add_argument('--exp_dir', type=str, default="./results", help='directory to save experiments')
        self.add_argument('--exp_type', type=str, default='sptemp', choices=['sp', 'sptemp'], help='experiment type')
        self.add_argument('--load_weights', type = bool, default = False, help = 'Load model weights or not')

        # data
        self.add_argument('--data_dir', type=str, default='AOD_data', help='directory to load data')
        self.add_argument('--ttsplit', type=float, default=0.10, help="train/test split for test and val")
        self.add_argument('--btrain', type=int, default= 1024, help='training batch size')
        self.add_argument('--bval', type=int, default= 1024, help='validation batch size')
        self.add_argument('--btest', type=int, default= 1024, help='testing batch size')
        self.add_argument('--load_data', type=bool, default= False, help='Load or generate new data')
        self.add_argument('--sample_rate', type=float, default= 0.10, help='Sample rate for AOD')
        self.add_argument('--test_time', type=float, default= 20, help='Test Date for AOD')
        self.add_argument('--temporal', type=bool, default= True, help="add temporal or not")
        self.add_argument('--n_dim', type=int, default=15000, help='number of sites')

        # model
        self.add_argument('--input_size', type=int, default=2, help='input dimension')
        self.add_argument('--aux_size', type=int, default= 1024, help='latent hidden dimension')
        self.add_argument('--latent_mod', type=str, default="FFNP", help='latent model type [ResMLP, FFNP, FFNG]')
        self.add_argument('--block_size', type=int, default= 5, help='number of latent layers')
        self.add_argument('--gp_size', type=int, default= 5000, help='full GP dimension')
        self.add_argument('--latent_size', type=int, default = 128, help='latent output dimension')
        self.add_argument('--output_size', type=int, default=1, help='output dimension')
        self.add_argument('--latent_temporal', type=bool, default=True, help='latent temporal model or not')
        self.add_argument('--mod_temporal', type=bool, default=True, help='full temporal model or not')
        self.add_argument('--nSVGD', type=int, default=10, help='number of model instances in SVGD')

        # prior
        self.add_argument('--nu_prior', type=list, default=[0.5, 0.5], help='Lognormal Prior parameters for Nu')
        self.add_argument('--rho_prior', type=list, default=[-2.0, 1.0], help='Gamma Prior parameters for Rho')
        self.add_argument('--rhot_prior', type=list, default=[-1.0, 0.5], help='Normal Prior parameters for Temporal Rho')
        self.add_argument('--rhol_prior', type=list, default=[-2.0, 1.0], help='Normal Prior parameters for Latent Rho')
        self.add_argument('--sig2_prior', type=list, default=[0.1, 0.1], help='InvGamma Prior parameters for Sig2')
        self.add_argument('--tau_prior', type=list, default=[0.1, 0.1], help='InvGamma Prior parameters for Tau')

        # training
        self.add_argument('--epochs', type=int, default= 15, help='number of epochs to train')
        self.add_argument('--lr', type=float, default=1e-5, help='ADAM learning rate GP layer')
        self.add_argument('--lr_latent', type=float, default=1e-5, help='ADAM learning rate Latent model')
        self.add_argument('--lr_noise', type=float, default=1e-3, help='ADAM learning rate Cov Params')
        self.add_argument('--lrs', type=str, default='ReduceLROnPlateau', help="learning rate scheduler")
        self.add_argument('--weight-decay', type=float, default=1e-5, help="weight decay")

        # conformal
        self.add_argument('--conformal', type=bool, default= False, help='should we use conformal predictions')
        self.add_argument('--conformal_neigh', type=list, default= [30, 80, 6], help='search range conformal nearest neighbors')
        self.add_argument('--conformal_grid', type=int, default=75, help='grid size of y')
        self.add_argument('--alpha', type=int, default=0.05, help='desired alpha level')
        self.add_argument('--conformal_parallel', type=bool, default=True, help='parallelize conformal evaluation')
        self.add_argument('--conformal_par_device', type=str, default='gpu', help='parallelize conformal over cpu or gpu')
        self.add_argument('--cores', type=int, default=10, help='Number of CPU cores')

        # logging
        self.add_argument('--log-freq', type=int, default=1, help='how many epochs to wait before logging training status')
        self.add_argument('--num-Plot', type=int, default=3, help='how many samples to plot')
        self.add_argument('--seed', type=int, default=123, help='manual seed used in PyTorch and Numpy')

    def parse(self):
        args = self.parse_args()

        args.save_dir = args.exp_dir + '/' + args.data_dir + '/' + args.latent_mod
            
        mkdir(args.save_dir)
        if args.seed is None:
            args.seed = random.randint(1, 10000)
        print("Random Seed: ", args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        args.gpu_devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None
        args.cpu_devices = [torch.device("cpu")]*args.cores
        return args


#####################################
# define arguments
#####################################
print('----------- Arguments -----------')
args = Parser().parse()
pprint(vars(args))
print('----------- End -----------\n')

if __name__ == '__main__':
    args = Parser().parse()
    print('-----------------------------------------')
    print('--------------- Arguments ---------------')
    pprint(vars(args))
    print('------------------ End ------------------')
    print('-----------------------------------------')    


