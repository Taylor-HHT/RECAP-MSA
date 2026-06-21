import torch
from torch import nn
import torch.nn.functional as F
import math
from .basic_layers import Transformer, CrossTransformer, HhyperLearningEncoder, GradientReversalLayer
from .bert import BertTextEncoder
from einops import rearrange, repeat
from .factorSSL import InfoNCECritic, CLUBInfoNCECritic, mlp_head


class RECAP(nn.Module):
    def __init__(self, args):
        super(RECAP, self).__init__()

        self.bs = args['base']['batch_size']
        self.num_modalities = 3
        self.token_len = args['model']['feature_extractor']['token_length'][0]  # 8
        self.hidden_dim = args['model']['feature_extractor']['hidden_dims'][0]  # 128
        self.adversarial_loss_fn = nn.BCELoss()
        self.final_pred_fc = nn.Linear(args['model']['fusion']['final_predictor']['hidden_dim'], 1)  # stage 2

        self.feat_dims = [self.token_len*self.hidden_dim, self.token_len*self.hidden_dim]  # 1024, 1024
        activation = 'relu'
        self.critic_hidden_dim = 2048
        self.critic_layers = 1
        temperature = 1
        
        # linear projection heads
        self.linears_infonce_x1x2 = nn.ModuleList([mlp_head(self.feat_dims[i], self.feat_dims[i]) for i in range(2)])
        self.linears_club_x1x2_cond = nn.ModuleList([mlp_head(self.feat_dims[i], self.feat_dims[i]) for i in range(2)])

        self.linears_infonce_x1y = mlp_head(self.feat_dims[0], self.feat_dims[0])
        self.linears_infonce_x2y = mlp_head(self.feat_dims[1], self.feat_dims[1])
        self.linears_infonce_x1x2_cond = nn.ModuleList([mlp_head(self.feat_dims[i], self.feat_dims[i]) for i in range(2)])
        self.linears_club_x1x2 = nn.ModuleList([mlp_head(self.feat_dims[i], self.feat_dims[i]) for i in range(2)])

        # critics
        self.infonce_x1x2 = InfoNCECritic(self.feat_dims[0], self.feat_dims[1], self.critic_hidden_dim, self.critic_layers, activation, temperature=temperature)
        self.club_x1x2_cond = CLUBInfoNCECritic(self.feat_dims[0]*2, self.feat_dims[1]*2, 
                                                self.critic_hidden_dim, self.critic_layers, activation, temperature=temperature)  

        self.infonce_x1y = InfoNCECritic(self.feat_dims[0], self.feat_dims[0], self.critic_hidden_dim, self.critic_layers, activation, temperature=temperature) 
        self.infonce_x2y = InfoNCECritic(self.feat_dims[1], self.feat_dims[1], self.critic_hidden_dim, self.critic_layers, activation, temperature=temperature) 
        self.infonce_x1x2_cond = InfoNCECritic(self.feat_dims[0]*2, self.feat_dims[1]*2, 
                                               self.critic_hidden_dim, self.critic_layers, activation, temperature=temperature) 
        self.club_x1x2 = CLUBInfoNCECritic(self.feat_dims[0], self.feat_dims[1], self.critic_hidden_dim, self.critic_layers, activation, temperature=temperature)

        self.bertmodel = BertTextEncoder(use_finetune=True, transformers='bert', pretrained=args['model']['feature_extractor']['bert_pretrained'])

        self.proj_l = nn.Sequential(
            nn.Linear(args['model']['feature_extractor']['input_dims'][0], args['model']['feature_extractor']['hidden_dims'][0]),
            Transformer(num_frames=args['model']['feature_extractor']['input_length'][0], 
                        save_hidden=False, 
                        token_len=args['model']['feature_extractor']['token_length'][0], 
                        dim=args['model']['feature_extractor']['hidden_dims'][0], 
                        depth=args['model']['feature_extractor']['depth'], 
                        heads=args['model']['feature_extractor']['heads'], 
                        mlp_dim=args['model']['feature_extractor']['hidden_dims'][0])
        )

        self.proj_a = nn.Sequential(
            nn.Linear(args['model']['feature_extractor']['input_dims'][2], args['model']['feature_extractor']['hidden_dims'][2]),
            Transformer(num_frames=args['model']['feature_extractor']['input_length'][2], 
                        save_hidden=False, 
                        token_len=args['model']['feature_extractor']['token_length'][2], 
                        dim=args['model']['feature_extractor']['hidden_dims'][2], 
                        depth=args['model']['feature_extractor']['depth'], 
                        heads=args['model']['feature_extractor']['heads'], 
                        mlp_dim=args['model']['feature_extractor']['hidden_dims'][2])
        )

        self.proj_v = nn.Sequential(
            nn.Linear(args['model']['feature_extractor']['input_dims'][1], args['model']['feature_extractor']['hidden_dims'][1]),
            Transformer(num_frames=args['model']['feature_extractor']['input_length'][1], 
                        save_hidden=False, 
                        token_len=args['model']['feature_extractor']['token_length'][1], 
                        dim=args['model']['feature_extractor']['hidden_dims'][1], 
                        depth=args['model']['feature_extractor']['depth'], 
                        heads=args['model']['feature_extractor']['heads'], 
                        mlp_dim=args['model']['feature_extractor']['hidden_dims'][1])
        )
        
        self.discriminator_v = Discriminator(args['model']['GAN']['discriminator']['input_dim'])
        self.discriminator_l = Discriminator(args['model']['GAN']['discriminator']['input_dim'])
        self.discriminator_a = Discriminator(args['model']['GAN']['discriminator']['input_dim'])

        self.modal_predictors = nn.ModuleList([
            nn.Sequential(
                Transformer(
                    num_frames=8,  # Keep consistent with token_len
                    save_hidden=False,
                    token_len=None,
                    dim=args['model']['fusion']['modal_predictors']['hidden_dim'],
                    depth=2,
                    heads=4,
                    mlp_dim=args['model']['fusion']['modal_predictors']['hidden_dim'] * 2
                ),
                nn.Linear(args['model']['fusion']['modal_predictors']['hidden_dim'], 
                        args['model']['fusion']['modal_predictors']['hidden_dim'] // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(args['model']['fusion']['modal_predictors']['hidden_dim'] // 2, 1)
            ) for _ in range(3)
        ])


        self.reconstructor = nn.ModuleList([
            Transformer(num_frames=args['model']['reconstructor']['input_length'], 
                        save_hidden=False, 
                        token_len=None, 
                        dim=args['model']['reconstructor']['input_dim'], 
                        depth=args['model']['reconstructor']['depth'], 
                        heads=args['model']['reconstructor']['heads'], 
                        mlp_dim=args['model']['reconstructor']['hidden_dim']) for _ in range(3)
        ])


        
        self.generator = nn.ModuleList([
            Transformer(num_frames=args['model']['reconstructor']['input_length'], 
                        save_hidden=False, 
                        token_len=None, 
                        dim=args['model']['reconstructor']['input_dim'], 
                        depth=args['model']['reconstructor']['depth'], 
                        heads=args['model']['reconstructor']['heads'], 
                        mlp_dim=args['model']['reconstructor']['hidden_dim']) for _ in range(3)
        ])


        self.attn_proj = nn.Linear(args['model']['fusion']['atten_projection']['hidden_dim'], 3 * args['model']['fusion']['atten_projection']['hidden_dim'])


    def compute_pairwise_ssl_loss(self, complete_feat_1, complete_feat_2, generated_feat_1, generated_feat_2):
        uncond_losses = [self.infonce_x1x2(self.linears_infonce_x1x2[0](complete_feat_1), self.linears_infonce_x1x2[1](complete_feat_2)),
                        self.club_x1x2(self.linears_club_x1x2[0](complete_feat_1), self.linears_club_x1x2[1](complete_feat_2)),
                        self.infonce_x1y(self.linears_infonce_x1y(complete_feat_1), self.linears_infonce_x1y(generated_feat_1)),
                        self.infonce_x2y(self.linears_infonce_x2y(complete_feat_2), self.linears_infonce_x2y(generated_feat_2))
        ]

        cond_losses = [self.infonce_x1x2_cond(torch.cat([self.linears_infonce_x1x2_cond[0](complete_feat_1), 
                                                        self.linears_infonce_x1x2_cond[0](generated_feat_1)], dim=1), 
                                            torch.cat([self.linears_infonce_x1x2_cond[1](complete_feat_2), 
                                                        self.linears_infonce_x1x2_cond[1](generated_feat_2)], dim=1)),
                    self.club_x1x2_cond(torch.cat([self.linears_club_x1x2_cond[0](complete_feat_1), 
                                                    self.linears_club_x1x2_cond[0](generated_feat_1)], dim=1), 
                                        torch.cat([self.linears_club_x1x2_cond[1](complete_feat_2), 
                                                    self.linears_club_x1x2_cond[1](generated_feat_2)], dim=1))
        ]   
        return uncond_losses, cond_losses


    def compute_adversarial_loss(self, x, generator, discriminator, num_segments_list=[8, 4, 2]):
        """
        Compute multi-scale adversarial loss, where each previous segment is used
        to adversarially generate the next segment.

        Args:
        x: Tensor with shape [bs, token_len, feature_dim]
        generator: generator model
        discriminator: discriminator model
        num_segments_list: list of temporal granularities to evaluate (default [8, 4, 2])

        Returns:
        losses: dict of adversarial losses for each temporal granularity
        """
        losses = {}
        device = x.device
        
        for num_segments in num_segments_list:
            segment_size = x.shape[1] // num_segments # Length of each segment
            loss = 0.0
            for i in range(num_segments - 1):  # Compare adjacent temporal segments
                x_prev = x[:, i * segment_size:(i + 1) * segment_size, :]  # Previous segment
                x_next = x[:, (i + 1) * segment_size:(i + 2) * segment_size, :] # Next adjacent segment

                # Compute neighborhood similarity with cosine similarity
                similarity = F.cosine_similarity(x_prev, x_next, dim=-1)  # [bs]
                # weight = similarity.unsqueeze(1)  # Shape [bs, 1]
                weight = ((similarity + 1) / 2).unsqueeze(1)  # Map similarity from [-1, 1] to [0, 1] as weights

                
                x_prev = x_prev.to(device)  
                x_next = x_next.to(device) 
                x_next_pred = generator(x_prev).to(device)  # Generator prediction
                real_label = torch.ones_like(discriminator(x_next))  # Match the discriminator output shape
                fake_label = torch.zeros_like(discriminator(x_next)) 
                
                # Discriminator loss
                real_loss = self.adversarial_loss_fn(discriminator(x_next), real_label)  # Classify the real next segment
                fake_loss = self.adversarial_loss_fn(discriminator(x_next_pred.detach()), fake_label)  # Classify the generated next segment
                disc_loss = (real_loss + fake_loss) / 2
                
                # Generator loss
                gen_loss = self.adversarial_loss_fn(discriminator(x_next_pred), real_label) * weight.mean()
                
                loss += gen_loss + disc_loss
            
            losses[num_segments] = loss
        
        return losses
        


    def forward(self, complete_input, incomplete_input, labels, mode="completion"):

        vision, audio, language = complete_input
        vision_m, audio_m, language_m = incomplete_input  # (bs, input_len, input_dim)
        
        h_1_v = self.proj_v(vision_m)[:, :8]
        h_1_a = self.proj_a(audio_m)[:, :8]  # (bs, 375, 5)--> (bs,8,128)
        h_1_l = self.proj_l(self.bertmodel(language_m))[:, :8]

        if mode == "completion":

            b = vision_m.size(0)

            losses_adv_v = self.compute_adversarial_loss(h_1_v, self.generator[2], self.discriminator_v, num_segments_list=[8, 4, 2])
            losses_adv_l = self.compute_adversarial_loss(h_1_l, self.generator[0], self.discriminator_l, num_segments_list=[8, 4, 2])
            losses_adv_a = self.compute_adversarial_loss(h_1_a, self.generator[1], self.discriminator_a, num_segments_list=[8, 4, 2])
            w_l, w_v, w_a = 0.5, 0.3, 0.2
            loss_adv = (
                w_v * sum(losses_adv_v.values()) +
                w_l * sum(losses_adv_l.values()) +
                w_a * sum(losses_adv_a.values())
            )            

            generated_vision = self.generator[2](h_1_v)  # (bs, 8, dim)
            generated_audio = self.generator[1](h_1_a)
            generated_language = self.generator[0](h_1_l)
            
            rec_feats, complete_feats = None, None

            uncond_losses_LV, cond_losses_LV = [], []
            uncond_losses_LA, cond_losses_LA = [], []
            if (vision is not None) and (audio is not None) and (language is not None):
                rec_feat_a = self.reconstructor[0](h_1_a)
                rec_feat_v = self.reconstructor[1](h_1_v)
                rec_feat_l = self.reconstructor[2](h_1_l)

                rec_feats = torch.cat([rec_feat_a, rec_feat_v, rec_feat_l], dim=1) 

                # Compute the complete features as the label of reconstruction
                complete_language_feat = self.proj_l(self.bertmodel(language))[:, :8]
                complete_vision_feat = self.proj_v(vision)[:, :8]
                complete_audio_feat = self.proj_a(audio)[:, :8]
            
                complete_feats = torch.cat([complete_audio_feat, complete_vision_feat, complete_language_feat], dim=1) # Reconstruction target features
            
                # Add the common-information part by treating vision_m, audio_m, and language_m
                # as views of vision, audio, and language, respectively
                complete_language_feat = complete_language_feat.view(b, -1)
                complete_vision_feat = complete_vision_feat.view(b, -1)
                complete_audio_feat = complete_audio_feat.view(b, -1)
                generated_language = generated_language.view(b, -1)
                generated_vision = generated_vision.view(b, -1)
                generated_audio = generated_audio.view(b, -1)

                uncond_losses_LV, cond_losses_LV = self.compute_pairwise_ssl_loss(complete_language_feat, complete_vision_feat, generated_language, generated_vision)
                uncond_losses_LA, cond_losses_LA = self.compute_pairwise_ssl_loss(complete_language_feat, complete_audio_feat, generated_language, generated_audio)
                uncond_losses_VA, cond_losses_VA = self.compute_pairwise_ssl_loss(complete_vision_feat, complete_audio_feat, generated_vision, generated_audio)

                complete_language_feat = complete_language_feat.view(b, -1, 128)
                complete_vision_feat = complete_vision_feat.view(b, -1, 128)
                complete_audio_feat = complete_audio_feat.view(b, -1, 128)
                generated_language = generated_language.view(b, -1, 128)
                generated_vision = generated_vision.view(b, -1, 128)
                generated_audio = generated_audio.view(b, -1, 128)

            return {
                "rec_feats": rec_feats,
                "complete_feats": complete_feats,
                "loss_adv": loss_adv,
                "Factorized_LV": sum(uncond_losses_LV) + sum(cond_losses_LV),
                "Factorized_LA": sum(uncond_losses_LA) + sum(cond_losses_LA),
                "Factorized_VA": sum(uncond_losses_VA) + sum(cond_losses_VA),
            }

        elif mode == "fusion_prediction":        
            
            # Use the trained generator for modality completion
            generated_language = self.generator[0](h_1_l)
            generated_audio = self.generator[1](h_1_a)
            generated_vision = self.generator[2](h_1_v)


            feats = torch.stack([generated_language, generated_audio, generated_vision], dim=1)  # (batch, 3, 8, hidden_size)
            
            # Compute the prediction output for each modality: (batch, 3, 8, 1)
            preds = torch.stack([self.modal_predictors[i](feats[:, i, :, :]) for i in range(3)], dim=1)  # (batch, 3, 8, 1)

            # # Estimate mutual information (MI) using MSE as a proxy: (batch, 3, 8)
            mi_scores = -F.mse_loss(preds.squeeze(-1), labels.unsqueeze(1).repeat(1, 3, 8), reduction='none')

            # # Average across the 8 tokens to obtain modality-level MI estimates: (batch, 3)
            mi_scores = mi_scores.mean(dim=-1)

            qkvs = self.attn_proj(feats)  # (batch, 3, 8, 3 * hidden_size) (64,3,8,128)
            q, v, k = qkvs.chunk(3, dim=-1)  # (batch, 3, 8, hidden_size)
            q_mean = q.mean(dim=2)  # (batch, 3, hidden_size)
            k_mean = k.mean(dim=2) 
            v_mean = v.mean(dim=2) 

            attn_logits = torch.einsum('bmd,bmd->bm', q_mean, k_mean)
            attn_logits = attn_logits / math.sqrt(q.shape[-1])

            attn_weights = F.softmax(attn_logits, dim=-1)  # (batch, 3)

            feat_final = torch.matmul(attn_weights.unsqueeze(1), v_mean).squeeze(1)

            # feat_final = feat_final.mean(dim=1)
            # feat_final = torch.mean(v_mean, dim=1)
            pred_final = self.final_pred_fc(feat_final)  # (batch, 1)
            
            ranking_loss = self.compute_ranking_loss(attn_weights, mi_scores)

            output = pred_final


            return {'sentiment_preds': output, 
                    'final_pred': pred_final,
                    'ranking_loss': ranking_loss}

    def compute_ranking_loss(self, attn_weights, mi_scores, margin=0.1):
        """
        Compute ranking loss so that modalities with higher mutual information
        receive larger attention weights.
        Args:
            attn_weights: (batch, 3) attention weights
            mi_scores: (batch, 3) mutual information estimates
            margin: (float) minimum margin between weights
        Returns:
            ranking_loss: (float) ranking loss
        """
        batch_size, num_modalities = attn_weights.shape
        loss = 0.0

        for i in range(num_modalities):
            for j in range(num_modalities):
                if i == j:
                    continue  # Skip self-comparison

                indicator = (mi_scores[:, i] > mi_scores[:, j]).float()  # 1 if mi_i > mi_j, else 0
                loss += F.margin_ranking_loss(
                    attn_weights[:, i], attn_weights[:, j], indicator, margin=margin, reduction='mean'
                )

        return loss / max(1, num_modalities * (num_modalities - 1))  # Normalize


class Generator(nn.Module):
    def __init__(self, feature_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
    
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, feature_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)



def build_model(args):
    return RECAP(args)
