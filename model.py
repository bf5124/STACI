import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from latent_models import FFNet, ResMLP

class RFF_Net(nn.Module):
    def __init__(self, input_size = 2, aux_size = 250, gp_size = 250, latent_mod = "ResMLP", n_blocks = 5, output_size = 1\
                 ,latent_size = 1, latent_temporal = False, mod_temporal = False, t_steps = None):
        super(RFF_Net, self).__init__()

        self.latent_temporal = latent_temporal
        
        if latent_temporal == True:
            latent_input_size = input_size + 1
        else:
            latent_input_size = input_size

            
        if latent_mod == "ResMLP":
            self.latent = ResMLP(
                                res_in_dim = latent_input_size,
                                res_out_dim = latent_size,
                                res_width = aux_size,
                                res_depth = n_blocks,
                                net_act = "gelu",)
        if latent_mod == "FFNP":
            self.latent = FFNet(
                                encode_method="Position",
                                gauss_sigma=4.0,
                                gauss_input_size= latent_input_size,
                                gauss_encoded_size= aux_size,
                                pos_freq_const= latent_input_size*10,
                                pos_freq_num= int(aux_size*1),
                                net_in= int(aux_size*1)*2*latent_input_size,
                                net_hidden=aux_size,
                                net_out=latent_size,
                                net_layers=n_blocks,
                                net_act="gelu",
                               )
        if latent_mod == "FFNG":
            self.latent = FFNet(
                                encode_method="Gaussian",
                                gauss_sigma=1.0,
                                gauss_input_size= latent_input_size,
                                gauss_encoded_size= aux_size,
                                pos_freq_const= latent_input_size*10,
                                pos_freq_num= int(aux_size*1),
                                net_in= 2*aux_size,
                                net_hidden=aux_size,
                                net_out=latent_size,
                                net_layers=n_blocks,
                                net_act="gelu",
                               )

        if mod_temporal == True:
            self.fc3 = nn.Linear(input_size + latent_size + 1, gp_size, bias=False)
        else:
            self.fc3 = nn.Linear(input_size + latent_size, gp_size, bias=False)

        self.fc4a = nn.Linear(gp_size, output_size, bias=False)
        self.fc4b = nn.Linear(gp_size, output_size, bias=False)


        torch.nn.init.normal_(self.fc4a.weight, mean=0.0, std=np.sqrt(1.0 / gp_size))
        torch.nn.init.normal_(self.fc4b.weight, mean=0.0, std=np.sqrt(1.0 / gp_size))

    def forward(self, x):

        if self.latent_temporal == False:
            L_out = self.latent(x[:, :2])
        else:
            L_out = self.latent(x)


        skip = torch.cat((x, L_out), dim=1)
        out = self.fc3(skip)
        out1 = torch.cos(out)
        out2 = torch.sin(out)

        out = self.fc4a(out1) + self.fc4b(out2)

        return out

    def _count_parameters(self):
        n_params = 0
        for name, param in self.named_parameters():
            print(name)
            print(param.size())
            print(param.numel())
            n_params += param.numel()
            print('num of parameters so far: {}'.format(n_params))

    def reset_parameters(self, verbose=False):
        for module in self.modules():
            # pass self, otherwise infinite loop
            if isinstance(module, self.__class__):
                continue
            if 'reset_parameters' in dir(module):
                if callable(module.reset_parameters):
                    module.reset_parameters()
                    if verbose:
                        print("Reset parameters in {}".format(module))


if __name__ == '__main__':
    RFF_ed = RFF_Net(input_size = 2, aux_size = 250, gp_size = 250, latent_mod = "ResMLP", n_blocks = 5, output_size = 1, mod_temporal = False, t_steps = None)
    print(RFF_ed)
    x = torch.Tensor(0.12, 0.45)
    print(RFF_ed._count_parameters())



