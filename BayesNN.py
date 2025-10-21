import torch
import torch.nn as nn
import math
import copy

from torch.nn.parameter import Parameter
from utilities import log_sum_exp


class BayesNet(nn.Module):
    def __init__(self, args, model):
        super(BayesNet, self).__init__()
        if not isinstance(model, nn.Module):
            raise TypeError("model {} is not a Module subclass".format(torch.typename(model)))
        self.n_samples = args.nSVGD
        self.device = args.device

        self.nu_prior_loc = args.nu_prior[0]
        self.nu_prior_scale = args.nu_prior[1]
        self.rho_prior_loc = args.rho_prior[0]
        self.rho_prior_scale = args.rho_prior[1]
        self.rhol_prior_loc = args.rhol_prior[0]
        self.rhol_prior_scale = args.rhol_prior[1]
        self.rhot_prior_loc = args.rhot_prior[0]
        self.rhot_prior_scale = args.rhot_prior[1]
        self.sig2_prior_loc = args.sig2_prior[0]
        self.sig2_prior_scale = args.sig2_prior[1]
        self.tau_prior_loc = args.tau_prior[0]
        self.tau_prior_scale = args.tau_prior[1]

        self.latent_size = args.latent_size

        instances = []
        for i in range(self.n_samples):
            new_instance = copy.deepcopy(model)
            new_instance.reset_parameters()
            print('Reset parameters in model instance {}'.format(i+1))
            instances.append(new_instance)

        self.nnets = nn.ModuleList(instances)
        del instances

        log_nu_Z = torch.distributions.LogNormal(self.nu_prior_loc, self.nu_prior_scale).sample((self.n_samples,)).to(
            self.device).log()
        
        log_rho_Z = torch.distributions.LogNormal(self.rho_prior_loc, self.rho_prior_scale).sample((self.n_samples,)).to(self.device).log()
        
        log_rhol_Z = torch.distributions.LogNormal(self.rho_prior_loc, self.rho_prior_scale).sample((self.n_samples,)).to(self.device).log()

        log_sig2_Z = torch.full((self.n_samples,), 0.5).log()
        log_tau = torch.full((self.n_samples,), 0.05).log()

        if args.temporal == True:

            log_rhot_Z = torch.distributions.LogNormal(self.rhot_prior_loc, self.rhot_prior_scale).sample((self.n_samples,)).to(self.device).log()

        for i in range(self.n_samples):
            self.nnets[i].log_nu_Z = Parameter(log_nu_Z[i])
            self.nnets[i].log_rho_Z = Parameter(log_rho_Z[i])
            self.nnets[i].log_rhol_Z = Parameter(log_rhol_Z[i])
            self.nnets[i].log_sig2_Z = Parameter(log_sig2_Z[i])
            self.nnets[i].log_tau = Parameter(log_tau[i])
            if args.temporal == True:
                self.nnets[i].log_rhot_Z = Parameter(log_rhot_Z[i])


        print('Total number of parameters: {}'.format(self._num_parameters()))

    def _num_parameters(self):
        count = 0
        for name, param in self.named_parameters():
            count += param.numel()
        return count

    def __getitem__(self, idx):
        return self.nnets[idx]


    @property
    def log_nu_Z(self):
        return torch.tensor([self.nnets[i].log_nu_Z.item()
            for i in range(self.n_samples)], device=self.device)


    @property
    def log_rho_Z(self):
        return torch.tensor([self.nnets[i].log_rhol_Z.item()
             for i in range(self.n_samples)], device=self.device)

    @property
    def log_rhol_Z(self):
        return torch.tensor([self.nnets[i].log_rho_Z.item()
             for i in range(self.n_samples)], device=self.device)


    @property
    def log_sig2_Z(self):
        return torch.tensor([self.nnets[i].log_sig2_Z.item()
            for i in range(self.n_samples)], device=self.device)

    @property
    def log_tau(self):
        return torch.tensor([self.nnets[i].log_tau.item()
             for i in range(self.n_samples)], device=self.device)

    @property
    def log_rhot_Z(self):
        return torch.tensor([self.nnets[i].log_rhot_Z.item()
                             for i in range(self.n_samples)], device=self.device)


    def forward(self, input):
        output = []
        for i in range(self.n_samples):
            out = self.nnets[i].forward(input)
            output.append(out)

        output = torch.stack(output)


        return output

    def _log_joint(self, index, output, target, ntrain):

        log_likelihood = ntrain / output.size(0) * (
                - 1/(2 * self.nnets[index].log_tau.exp())
                * (target - output).pow(2).sum()
                - (target.numel()* self.nnets[index].log_tau)/2)


        # Remove Prior smoothess/Range, just change to regular normal or t prior on weights

        log_prob_prior_w = torch.tensor(0.).to(self.device)

        for param in self.nnets[index].latent.parameters():
            log_prob_prior_w += \
                torch.log1p(0.5 / 1 * param.pow(2)).sum()
        log_prob_prior_w *= -(0.05 + 0.5)



        log_prob_prior_omega = torch.tensor(0.).to(self.device)
        nu_Z = self.nnets[index].log_nu_Z
        rho_Z = self.nnets[index].log_rho_Z
        rhol_Z = self.nnets[index].log_rhol_Z
        J_dim = self.nnets[index].fc4a.in_features

        for param in self.nnets[index].fc3.parameters():
            if hasattr(self.nnets[index], 'log_rhot_Z'):

                rhot_Z = self.nnets[index].log_rhot_Z

                range_spatial = (1 / (4.0 / ((torch.pi * rho_Z.exp()).pow(2)))).item()
                range_temporal = (1 / (4.0 / ((torch.pi * rhot_Z.exp()).pow(2)))).item()
                range_latent = (1 / (4.0 / ((torch.pi * rhol_Z.exp()).pow(2)))).item()
                
                block1 = torch.diag(torch.tensor([range_spatial, range_spatial], dtype=torch.float32))
                block2 = torch.diag(torch.tensor([range_temporal], dtype=torch.float32))
                block3 = torch.diag(torch.full((self.latent_size,), range_latent, dtype=torch.float32))
                
                # Combine them into a block diagonal matrix.
                omega_cov = torch.block_diag(block1, block2, block3).to(self.device)

                log_prob_prior_omega += \
                    (torch.special.gammaln(0.5 * (2 * nu_Z.exp() + 4.0)) - torch.special.gammaln(0.5 * (2 * nu_Z.exp())) \
                     - 0.5 * torch.logdet(torch.inverse(omega_cov)) - 2.0 * torch.log(nu_Z.exp() * math.pi) \
                     - (nu_Z.exp() + 2.0) * torch.log1p(
                                (1 / (2 * nu_Z.exp())) * (param @ omega_cov * param).sum(
                                    dim=1))).sum()

                del rhot_Z, block1, block2, block3
            else:
                range_spatial = (1 / (4.0 / ((torch.pi * rho_Z.exp()).pow(2)))).item()
                range_latent = (1 / (4.0 / ((torch.pi * rhol_Z.exp()).pow(2)))).item()

                block1 = torch.diag(torch.tensor([range_spatial, range_spatial], dtype=torch.float32))
                block2 = torch.diag(torch.full((self.latent_size,), range_latent, dtype=torch.float32))
                
                omega_cov = torch.block_diag(block1, block2).to(self.device)

                log_prob_prior_omega += \
                    (torch.special.gammaln(0.5 * (2 * nu_Z.exp() + 3)) - torch.special.gammaln(0.5 * (2 * nu_Z.exp())) \
                     - 0.5 * torch.logdet(torch.inverse(omega_cov)) - 1.5 * torch.log(nu_Z.exp()*math.pi)\
                     -(nu_Z.exp() + 1.5) * torch.log1p((1 / (2 * nu_Z.exp())) * (param @ omega_cov * param).sum(
                                dim=1))).sum()

        log_prob_prior_nu_Z = (-1 / (2 * self.nu_prior_scale)) * (nu_Z - self.nu_prior_loc).pow(2).sum() - nu_Z

         #log_prob_prior_rho_Z = self.rho_prior_loc * rho_Z - rho_Z.exp() * self.rho_prior_scale
        log_prob_prior_rho_Z = (-1 / (2 * self.rho_prior_scale)) * (rho_Z - self.rho_prior_loc).pow(2).sum() - rho_Z

        log_prob_prior_rhol_Z = (-1 / (2 * self.rhol_prior_scale)) * (rhol_Z - self.rhol_prior_loc).pow(2).sum() - rhol_Z

        timebool = False
        if hasattr(self.nnets[index], 'log_rhot_Z'):
            timebool = True
            
            rhot_Z = self.nnets[index].log_rhot_Z
             #log_prob_prior_rhot_Z = self.rho_prior_loc * rhot_Z - rhot_Z.exp() * self.rho_prior_scale
            log_prob_prior_rhot_Z = (-1 / (2 * self.rhot_prior_scale)) * (rhot_Z - self.rhot_prior_loc).pow(2).sum() - rhot_Z


            del rhot_Z


        del omega_cov, nu_Z, rho_Z, rhol_Z

        log_prob_prior_ab = torch.tensor(0.).to(self.device)
        sig2_Z = self.nnets[index].log_sig2_Z
        tau = self.nnets[index].log_tau

        for param in self.nnets[index].fc4a.parameters():
            log_prob_prior_ab += \
                (-0.5*torch.log((sig2_Z.exp()/J_dim)) - param.pow(2)  / (2 * (sig2_Z.exp()/J_dim))).sum()

        for param in self.nnets[index].fc4b.parameters():
            log_prob_prior_ab += \
                (-0.5*torch.log((sig2_Z.exp()/J_dim)) - param.pow(2)  / (2 * (sig2_Z.exp()/J_dim))).sum()

        log_prob_prior_sig2_Z = -self.sig2_prior_loc * torch.log(sig2_Z.exp()/J_dim)   - self.sig2_prior_scale / (sig2_Z.exp()/J_dim)
        log_prob_prior_tau = -self.tau_prior_loc * tau - self.tau_prior_scale / (tau.exp())
        del sig2_Z, tau, J_dim


        tot_like = log_likelihood + log_prob_prior_w\
                   + log_prob_prior_omega + log_prob_prior_ab \
                   + log_prob_prior_sig2_Z + log_prob_prior_tau \
                   + log_prob_prior_nu_Z + log_prob_prior_rho_Z + log_prob_prior_rhol_Z \

        if timebool == True:
            tot_like += log_prob_prior_rhot_Z


        return tot_like

    def _compute_mse_nlp(self, input, target, size_average=True, out=False):
        output = self.forward(input)

        log_tau = self.log_tau.unsqueeze(-1).unsqueeze(-1)
        log_2pi_S = torch.tensor(0.5 * target[0].numel() * math.log(2 * math.pi)
                                 + math.log(self.n_samples), device=self.device)

        exponent = - 0.5 * (1/log_tau.exp() * ((target - output) ** 2)).view(
            self.n_samples, target.size(0), -1).sum(-1) \
                   + 0.5 * target[0].numel() * 1/self.log_tau.unsqueeze(-1)

        nlp = - log_sum_exp(exponent, dim=0).mean() + log_2pi_S
        mse = ((target - output.mean(0)) ** 2).mean()

        if not size_average:
            mse *= target.numel()
            nlp *= target.size(0)
        if not out:
            return mse, nlp
        else:
            return mse, nlp, output

    def predict(self, x_test):
        y = self.forward(x_test)
        #y_pred_mean = y.mean(0)

        return y #y_pred_mean

    def propagate(self, mc_loader):
        output_size = mc_loader.dataset[0][1].size()
        cond_Ey = torch.zeros(self.n_samples, *output_size, device=self.device)
        cond_Eyy = torch.zeros_like(cond_Ey)

        for _, (x_mc, _) in enumerate(mc_loader):
            x_mc = x_mc.to(self.device)
            y, l = self.forward(x_mc)
            cond_Ey += y.mean(1)
            cond_Eyy += y.pow(2).mean(1)
        cond_Ey /= len(mc_loader)
        cond_Eyy /= len(mc_loader)
        tau_inv = (self.log_tau).exp()
        print('Noise variances: {}'.format(tau_inv))

        y_cond_pred_var = cond_Eyy - cond_Ey ** 2 \
                          + tau_inv.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        return cond_Ey.mean(0), cond_Ey.var(0), \
            y_cond_pred_var.mean(0), y_cond_pred_var.var(0)






