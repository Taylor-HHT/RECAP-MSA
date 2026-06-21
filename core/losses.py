from torch import nn
from torch.nn import functional as F
import torch

class MultimodalLoss_stage1(nn.Module):
    def __init__(self, args):
        super().__init__()
        # adv loss, SSCL loss, recon loss
        self.adv = args['base']['adv']
        self.recon = args['base']['recon']
        self.paraSSL1 = args['base']['paraSSL_LV']
        self.paraSSL2 = args['base']['paraSSL_LA']
        self.paraSSL3 = args['base']['paraSSL_VA']
        self.MSE_Fn = nn.MSELoss() 


    def forward(self, out, label):
        # adv loss, SSCL loss, recon loss
        l_rec = self.MSE_Fn(out['rec_feats'], out['complete_feats']) if out['rec_feats'] is not None and out['complete_feats'] is not None else 0

        l_ssl_LV = max(out['Factorized_LV'], -5.0)
        l_ssl_LA = max(out['Factorized_LA'], -5.0)
        l_ssl_VA = max(out['Factorized_VA'], -5.0)

        l_adv = max(out['loss_adv'], -3.0)
        loss = self.recon * l_rec  + self.adv * l_adv + self.paraSSL1 * l_ssl_LV + self.paraSSL2 * l_ssl_LA + self.paraSSL3 * l_ssl_VA

        return {'loss': loss, 'l_rec': l_rec, 'l_adv': l_adv, 'l_ssl_LV': l_ssl_LV, 'l_ssl_LA': l_ssl_LA, 'l_ssl_VA': l_ssl_VA}


class MultimodalLoss_stage2(nn.Module):
    def __init__(self, args):
        super().__init__()
        # fusion, prediction loss
        self.task = args['base']['task'] 
        self.para_rank = args['base']['para_rank']
        self.MSE_Fn = nn.MSELoss() 


    def forward(self, out, label):
        l_sp = self.MSE_Fn(out['sentiment_preds'], label['sentiment_labels'])

        l_ranking = out['ranking_loss']

        loss = self.task * l_sp + self.para_rank * l_ranking

        return {'loss': loss, 'l_sp': l_sp, 'ranking': l_ranking}
