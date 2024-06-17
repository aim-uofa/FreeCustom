import os,sys,datetime
from omegaconf import OmegaConf

import torch
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from pytorch_lightning import seed_everything

from diffusers import DDIMScheduler

from utils.utils import load_image, load_mask
from pipelines.pipeline_stable_diffusion_freecustom import StableDiffusionFreeCustomPipeline
from freecustom.mrsa import MultiReferenceSelfAttention
from freecustom.hack_attention import hack_self_attention_to_mrsa

if __name__ == "__main__":
    sys.path.append(os.getcwd())

    # load config
    cfg = OmegaConf.load("configs/config_stable_diffusion.yaml")
    print(f'config: {cfg}')

    # set results and log output root directory
    date = datetime.datetime.now().strftime("%Y%m%d")
    now  = datetime.datetime.now().strftime("%H-%M-%S")
    results_dir = os.path.join('results', date, now)

    # set device
    torch.cuda.set_device(cfg.gpu)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # set model
    scheduler = DDIMScheduler(  beta_start=0.00085, 
                                beta_end=0.012, 
                                beta_schedule="scaled_linear", 
                                clip_sample=False, 
                                set_alpha_to_one=False
                            )
    model = StableDiffusionFreeCustomPipeline.from_pretrained(cfg.model_path, scheduler=scheduler).to(device)
    model.safety_checker = None
    
    # prapare data (including prompt, mask, image, latent)
    ref_masks       = []
    ref_images      = []
    ref_prompts     = []
    ref_latents_z_0 = []
    for ref_image_info in cfg.ref_image_infos.items():
        ref_image_path  = ref_image_info[0]
        ref_text_prompt = ref_image_info[1]
        ref_mask_path   = ref_image_path.replace('/image/', '/mask/')
        ref_mask  = load_mask(ref_mask_path, device)
        ref_image = load_image(ref_image_path, device)
        ref_masks.append(ref_mask)
        ref_images.append(ref_image)
        ref_prompts.append(ref_text_prompt)
        ref_latents_z_0.append(model.image2latent(ref_image))
    
    # set prompt
    target_prompt = cfg.target_prompt
    if cfg.use_null_ref_prompts:  
        prompts = [target_prompt] + ([""] * (len(ref_prompts))) 
    else:
        prompts = [target_prompt] + ref_prompts
    negative_prompts = [cfg.negative_prompt] * len(prompts)

    # set dirs
    concepts_name  = ref_image_path.split('/')[3]
    results_dir    = os.path.join(results_dir + f" {concepts_name} \"" + target_prompt + f"\" {cfg.mark}")
    image_save_dir = os.path.join(results_dir, 'ref_images')
    mask_save_dir  = os.path.join(results_dir, 'ref_masks')
    os.makedirs(image_save_dir, exist_ok=True)
    os.makedirs(mask_save_dir, exist_ok=True)

    # set config for visualization
    viz_cfg = OmegaConf.load("configs/config_for_visualization.yaml")
    viz_cfg.results_dir = results_dir
    viz_cfg.ref_image_infos = cfg.ref_image_infos

    # save image, mask, and config
    OmegaConf.save(cfg, os.path.join(results_dir, "config.yaml"))
    for i, (ref_image, ref_mask) in enumerate(zip(ref_images, ref_masks)):
        save_image(ref_image*0.5+0.5, os.path.join(image_save_dir, f'image_{i}.png'))
        save_image(ref_mask.float(),  os.path.join(mask_save_dir, f'mask_{i}.png'))

    # run each seed
    for seed in cfg.seeds:
        seed_everything(seed)

        # hack the attention module
        mrsa = MultiReferenceSelfAttention(
                                start_step     = cfg.start_step,
                                end_step       = cfg.end_step,
                                layer_idx      = cfg.layer_idx, 
                                ref_masks      = ref_masks,
                                mask_weights   = cfg.mask_weights,
                                style_fidelity = cfg.style_fidelity,
                                viz_cfg        = viz_cfg)
        hack_self_attention_to_mrsa(model, mrsa)

        # set latent
        randn_latent_z_T = torch.randn_like(ref_latents_z_0[0])   # Initialize Gaussian noise for generated image $z_T$
        latents = torch.cat([randn_latent_z_T] + ref_latents_z_0) # Concatenate $z_T$ and the latent code of the reference images $z_0^'$
        
        # run freecustom
        images = model(
                    prompt=prompts,
                    latents=latents,
                    guidance_scale=7.5,
                    negative_prompt=negative_prompts,
                    ).images[0]
        images.save(os.path.join(results_dir, f"freecustom_{seed}.png"))
        
        # concat input images and generated image
        out_image = torch.cat([ref_image * 0.5 + 0.5 for ref_image in ref_images] + [ToTensor()(images).to(device).unsqueeze(0)], dim=0)
        save_image(out_image, os.path.join(results_dir, f"all_{seed}.png"))