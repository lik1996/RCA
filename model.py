import torch
import torch.nn as nn
from einops import rearrange, repeat
from .layers import GuideDecoder
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.upsample import SubpixelUpsample
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import math



class BERTModel(nn.Module):

    def __init__(self, bert_type, project_dim):

        super(BERTModel, self).__init__()

        self.model = AutoModel.from_pretrained(bert_type,output_hidden_states=True,trust_remote_code=True)
        self.project_head = nn.Sequential(             
            nn.Linear(768, project_dim),
            nn.LayerNorm(project_dim),             
            nn.GELU(),             
            nn.Linear(project_dim, project_dim)
        )
        # freeze the parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):

        output = self.model(input_ids=input_ids, attention_mask=attention_mask,output_hidden_states=True,return_dict=True)
        # get 1+2+last layer
        last_hidden_states = torch.stack([output['hidden_states'][1], output['hidden_states'][2], output['hidden_states'][-1]]) # n_layer, batch, seqlen, emb_dim
        embed = last_hidden_states.permute(1,0,2,3).mean(2).mean(1) # pooling
        embed = self.project_head(embed)

        return {'feature':output['hidden_states'],'project':embed}

class VisionModel(nn.Module):

    def __init__(self, vision_type, project_dim):
        super(VisionModel, self).__init__()

        self.model = AutoModel.from_pretrained(vision_type,output_hidden_states=True)   
        self.project_head = nn.Linear(768, project_dim)
        self.spatial_dim = 768

    def forward(self, x):

        output = self.model(x, output_hidden_states=True)
        embeds = output['pooler_output'].squeeze()
        project = self.project_head(embeds)

        return {"feature":output['hidden_states'], "project":project}


class RoleConditionedBlock(nn.Module):
    """Implements FEUP -> RCI -> FF and fuses with a previous encoder feature."""
    def __init__(self, in_channels, skip_channels, token_dim=768, hidden_d=64, n_queries=4):
        super().__init__()
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.token_dim = token_dim
        self.hidden_d = hidden_d
        self.n_queries = n_queries

        # FEUP: depthwise laplacian init
        self.hpf = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        lap = torch.tensor([[-1/8., -1/8., -1/8.],[-1/8., 1., -1/8.],[-1/8., -1/8., -1/8.]])
        w = lap.unsqueeze(0).unsqueeze(0).repeat(in_channels,1,1,1)
        with torch.no_grad():
            self.hpf.weight.copy_(w)

        # Conv block to produce F_com
        self.fe_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # FFN after residual
        self.fe_ffn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )

        # RCI components
        self.w_g = nn.Linear(token_dim, in_channels)
        att_dim = hidden_d
        self.W_l_sem = nn.Linear(token_dim, att_dim)
        self.W_v_sem = nn.Linear(in_channels, att_dim)

        self.queries = nn.Parameter(torch.randn(n_queries, att_dim))
        self.W_quan = nn.Linear(token_dim, att_dim)
        self.proj_quan_gate = nn.Linear(token_dim, att_dim)

        self.k_proj = nn.Linear(in_channels, att_dim)
        self.v_proj = nn.Linear(in_channels, att_dim)

        self.W_loc = nn.Linear(token_dim, att_dim)
        self.coord_mlp = nn.Sequential(nn.Linear(2, att_dim), nn.ReLU(), nn.Linear(att_dim, att_dim))
        self.alpha = nn.Parameter(torch.tensor(1.0))

        # fusion (FF) and output projection to match skip features
        self.agg = nn.Sequential(
            nn.Conv2d(in_channels*3, in_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )
        self.output_proj = nn.Conv2d(in_channels, self.skip_channels, kernel_size=1)

    def feup(self, x):
        hf = self.hpf(x)
        f_com = self.fe_conv(hf)
        x_mod = f_com * x + x
        return self.fe_ffn(x_mod)

    def rci(self, x, text_tokens):
        B,C,H,W = x.shape
        x_flat = rearrange(x, 'B C H W -> B (H W) C')
        text_pool = text_tokens.mean(dim=1)
        gate = torch.sigmoid(self.w_g(text_pool)).unsqueeze(1)
        x_glo = x_flat * gate

        l_proj = self.W_l_sem(text_tokens)
        v_proj = self.W_v_sem(x_glo)
        scores = torch.matmul(l_proj, v_proj.transpose(-2,-1)) / math.sqrt(self.hidden_d)
        att_map = torch.softmax(scores.mean(dim=1), dim=-1)
        f_sem = x_glo * att_map.unsqueeze(-1)

        shift = self.W_quan(text_pool).unsqueeze(1)
        q = self.queries.unsqueeze(0).repeat(B,1,1)
        gate_q = torch.sigmoid(self.proj_quan_gate(text_pool)).unsqueeze(1)
        q_prime = (q + shift) * gate_q

        K = self.k_proj(x_flat)
        V = self.v_proj(x_flat)
        scores_q = torch.matmul(q_prime, K.transpose(-2,-1)) / math.sqrt(self.hidden_d)
        att_q = torch.softmax(scores_q, dim=-1)
        f_quan = x_glo * att_q.mean(dim=1).unsqueeze(-1)

        coords = self._get_coords(H, W, x.device)
        P = self.coord_mlp(coords)
        loc_proj = self.W_loc(text_tokens).mean(dim=1)
        bias = torch.matmul(loc_proj.unsqueeze(1), P.transpose(0,1).unsqueeze(0).repeat(B,1,1)).squeeze(1)

        Q_loc = self.W_l_sem(text_tokens).mean(dim=1).unsqueeze(1)
        scores_loc = torch.matmul(Q_loc, K.transpose(-2,-1)).squeeze(1) / math.sqrt(self.hidden_d)
        scores_loc = scores_loc + self.alpha * bias
        att_loc = torch.softmax(scores_loc, dim=-1)
        f_loc = x_glo * att_loc.unsqueeze(-1)
        f_loc = torch.tanh(f_loc) * f_loc

        f_sem = rearrange(f_sem, 'B (H W) C -> B C H W', H=H, W=W)
        f_quan = rearrange(f_quan, 'B (H W) C -> B C H W', H=H, W=W)
        f_loc = rearrange(f_loc, 'B (H W) C -> B C H W', H=H, W=W)
        return f_sem, f_quan, f_loc

    def _get_coords(self, H, W, device):
        ys = torch.linspace(-1,1,steps=H,device=device)
        xs = torch.linspace(-1,1,steps=W,device=device)
        grid = torch.stack(torch.meshgrid(ys,xs), dim=-1)
        return rearrange(grid, 'h w c -> (h w) c')

    def forward(self, x_low, skip_feature, text_tokens):
        # Upsample lower-resolution feature to skip feature spatial size
        skip_H, skip_W = skip_feature.shape[-2:]
        x_up = F.interpolate(x_low, size=(skip_H, skip_W), mode='bilinear', align_corners=False)
        x_fe = self.feup(x_up)

        f_sem, f_quan, f_loc = self.rci(x_fe, text_tokens)
        fused = self.agg(torch.cat([f_sem, f_quan, f_loc], dim=1))
        fused = self.output_proj(fused)
        return fused + skip_feature


class LanGuideMedSeg(nn.Module):

    def __init__(self, bert_type, vision_type, project_dim=512):

        super(LanGuideMedSeg, self).__init__()

        self.encoder = VisionModel(vision_type, project_dim)
        self.text_encoder = BERTModel(bert_type, project_dim)

        self.spatial_dim = [7,14,28,56]    # 224*224
        feature_dim = [768,384,192,96]

        self.decoder16 = GuideDecoder(feature_dim[0],feature_dim[1],self.spatial_dim[0],24)
        self.decoder8 = GuideDecoder(feature_dim[1],feature_dim[2],self.spatial_dim[1],12)
        self.decoder4 = GuideDecoder(feature_dim[2],feature_dim[3],self.spatial_dim[2],9)
        self.decoder1 = SubpixelUpsample(2,feature_dim[3],24,4)
        self.out = UnetOutBlock(2, in_channels=24, out_channels=1)
        
        # Role-Conditioned Blocks: two instances; rcb1 will be reused at two scales
        # RCBs: process f4->f3, then f3->f2, then f2->f1
        self.rcb1 = RoleConditionedBlock(in_channels=feature_dim[0],
            skip_channels=feature_dim[1],
            token_dim=768,
            hidden_d=64,
            n_queries=4)
        self.rcb2 = RoleConditionedBlock(in_channels=feature_dim[1],
            skip_channels=feature_dim[2],
            token_dim=768,
            hidden_d=64,
            n_queries=4)
        self.rcb3 = RoleConditionedBlock(in_channels=feature_dim[2],
            skip_channels=feature_dim[3],
            token_dim=768,
            hidden_d=64,
            n_queries=4)

    def forward(self, data):

        image, text = data
        if image.shape[1] == 1:   
            image = repeat(image,'b 1 h w -> b c h w',c=3)

        image_output = self.encoder(image)
        image_features, image_project = image_output['feature'], image_output['project']
        text_output = self.text_encoder(text['input_ids'],text['attention_mask'])
        text_embeds, text_project = text_output['feature'],text_output['project']

        if len(image_features[0].shape) == 4: 
            image_features = image_features[1:]  # 4 8 16 32   convnext: Embedding + 4 layers feature map
            image_features = [rearrange(item,'b c h w -> b (h w) c') for item in image_features] 

        # 1) first RCB: align f4 -> fuse into f3, then use fused f3 as skip for decoder16
        os32 = image_features[3]  # f4 (B, N, C4)
        os32_map = rearrange(os32, 'B (H W) C -> B C H W', H=self.spatial_dim[0], W=self.spatial_dim[0])
        skip_f3 = rearrange(image_features[2], 'B (H W) C -> B C H W', H=self.spatial_dim[1], W=self.spatial_dim[1])
        fused_f3 = self.rcb1(os32_map, skip_f3, text_embeds[-1])
        # decoder16: use original os32 as vis, fused_f3 as skip
        os16 = self.decoder16(os32, rearrange(fused_f3, 'B C H W -> B (H W) C'), text_embeds[-1])

        # 2) second RCB: align os16 -> fuse into f2, then use fused f2 as skip for decoder8
        os16_map = rearrange(os16, 'B (H W) C -> B C H W', H=self.spatial_dim[1], W=self.spatial_dim[1])
        skip_f2 = rearrange(image_features[1], 'B (H W) C -> B C H W', H=self.spatial_dim[2], W=self.spatial_dim[2])
        fused_f2 = self.rcb2(os16_map, skip_f2, text_embeds[-1])
        os8 = self.decoder8(os16, rearrange(fused_f2, 'B C H W -> B (H W) C'), text_embeds[-1])

        # 3) final RCB: align os8 -> fuse into f1, then use fused f1 as skip for decoder4
        os8_map = rearrange(os8, 'B (H W) C -> B C H W', H=self.spatial_dim[2], W=self.spatial_dim[2])
        skip_f1 = rearrange(image_features[0], 'B (H W) C -> B C H W', H=self.spatial_dim[3], W=self.spatial_dim[3])
        fused_f1 = self.rcb3(os8_map, skip_f1, text_embeds[-1])
        os4 = self.decoder4(os8, rearrange(fused_f1, 'B C H W -> B (H W) C'), text_embeds[-1])
        os4_map = rearrange(os4, 'B (H W) C -> B C H W', H=self.spatial_dim[3], W=self.spatial_dim[3])
        os1 = self.decoder1(os4_map)

        out = self.out(os1).sigmoid()

        return out
    
