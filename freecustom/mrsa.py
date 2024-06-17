
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from utils.utils import visualize_correspondence, visualize_attention_map

class MultiReferenceSelfAttention():
    def __init__(self,  start_step=0, end_step=50, step_idx=None, layer_idx=None, ref_masks=None, mask_weights=[1.0,1.0,1.0], style_fidelity=1, viz_cfg=None):
        """
        Args:
            start_step   : the step to start transforming self-attention to multi-reference self-attention
            end_step     : the step to end transforming self-attention to multi-reference self-attention
            step_idx     : list of the steps to transform self-attention to multi-reference self-attention
            layer_idx    : list of the layers to transform self-attention to multi-reference self-attention
            ref_masks    : masks of the input reference images
            mask_weights : mask weights for each reference masks
            viz_cfg      : config for visualization
        """
        self.cur_step       =  0
        self.num_att_layers = -1
        self.cur_att_layer  =  0

        self.start_step   = start_step
        self.end_step     = end_step
        self.step_idx     = step_idx if step_idx is not None else list(range(start_step, end_step))
        self.layer_idx    = layer_idx
        
        self.ref_masks    = ref_masks
        self.mask_weights = mask_weights
        
        self.style_fidelity = style_fidelity

        self.viz_cfg = viz_cfg
       
    def __call__(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        out = self.mrsa_forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
        return out
    
    def get_ref_mask(self, ref_mask, mask_weight, H, W):
        ref_mask = ref_mask.float() * mask_weight
        ref_mask = F.interpolate(ref_mask, (H, W))
        ref_mask = ref_mask.flatten()
        return ref_mask
    
    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        B = q.shape[0] // num_heads
        H = W = int(np.sqrt(q.shape[1]))
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads) 
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        
        if kwargs.get("attn_batch_type") == 'mrsa':
            sim_own, sim_refs = sim[..., :H*W], sim[..., H*W:]
            sim_or = [sim_own]
            for i, (ref_mask, mask_weight) in enumerate(zip(self.ref_masks, self.mask_weights)):
                ref_mask = self.get_ref_mask(ref_mask, mask_weight, H, W)
                sim_ref = sim_refs[..., H*W*i: H*W*(i+1)]
                sim_ref = sim_ref + ref_mask.masked_fill(ref_mask == 0, torch.finfo(sim.dtype).min)
                sim_or.append(sim_ref)
            sim = torch.cat(sim_or, dim=-1)
        attn = sim.softmax(-1)
        
        # viz attention map within MRSA module
        if self.viz_cfg.viz_attention_map == True and \
            kwargs.get("attn_batch_type") == 'mrsa' and \
            self.cur_step in self.viz_cfg.viz_map_at_step and \
            self.cur_att_layer // 2 in self.viz_cfg.viz_map_at_layer:
            visualize_attention_map(attn, self.viz_cfg, self.cur_step, self.cur_att_layer//2)
        
        # viz feature correspondence within MRSA module
        if self.viz_cfg.viz_feature_correspondence == True and \
            kwargs.get("attn_batch_type") == 'mrsa' and \
            self.cur_step in self.viz_cfg.viz_corr_at_step and \
            self.cur_att_layer // 2 in self.viz_cfg.viz_corr_at_layer:
            visualize_correspondence(self.viz_cfg, attn, self.cur_step, self.cur_att_layer//2)
             
        if len(attn) == 2 * len(v):
            v = torch.cat([v] * 2)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=B, h=num_heads)
        return out  

    def sa_forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Original self-attention forward function
        """
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)
        return out
    
    def mrsa_forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Mutli-reference self-attention(MRSA) forward function
        """
        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return self.sa_forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        
        B = q.shape[0] // num_heads // 2
        
        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)

        # The first batch is the q,k,v feature of $z_t$ (own feature), and the subsequent batches are the q,k,v features of $z_t^'$ (reference featrue)
        qu_o, qu_r = qu[:num_heads], qu[num_heads:] 
        qc_o, qc_r = qc[:num_heads], qc[num_heads:]
        
        ku_o, ku_r = ku[:num_heads], ku[num_heads:]
        kc_o, kc_r = kc[:num_heads], kc[num_heads:]
        
        vu_o, vu_r = vu[:num_heads], vu[num_heads:]
        vc_o, vc_r = vc[:num_heads], vc[num_heads:]
        
        ku_cat, vu_cat = torch.cat([ku_o, *ku_r.chunk(B-1)], 1), torch.cat([vu_o, *vu_r.chunk(B-1)], 1)
        kc_cat, vc_cat = torch.cat([kc_o, *kc_r.chunk(B-1)], 1), torch.cat([vc_o, *vc_r.chunk(B-1)], 1)

        out_u_target = self.attn_batch(qu_o, ku_cat, vu_cat, None, None, is_cross, place_in_unet, num_heads, attn_batch_type='mrsa', **kwargs)
        out_c_target = self.attn_batch(qc_o, kc_cat, vc_cat, None, None, is_cross, place_in_unet, num_heads, attn_batch_type='mrsa', **kwargs)
        
        # The larger the style_fidelity, the more like the reference concepts, range of values: [0,1]
        if self.style_fidelity > 0:
            out_u_target = (1 - self.style_fidelity) * out_u_target + self.style_fidelity * self.attn_batch(qu_o, ku_o, vu_o, None, None, is_cross, place_in_unet, num_heads, **kwargs)

        out = self.sa_forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        out_u, out_c = out.chunk(2)
        out_u_ref, out_c_ref = out_u[1:], out_c[1:]
        out = torch.cat([out_u_target, out_u_ref, out_c_target, out_c_ref], dim=0)
        
        return out