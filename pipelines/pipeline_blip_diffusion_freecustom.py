from typing import List, Optional, Union

import PIL.Image
from PIL import Image

import torch
import numpy as np


from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from diffusers.pipelines.blip_diffusion.pipeline_blip_diffusion import BlipDiffusionPipeline

class BlipDiffusionFreeCustomPipeline(BlipDiffusionPipeline):
    
    @torch.no_grad()
    def image2latent(self, image):
        """
        Add this function to each self-defined pipeline
        """
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if type(image) is Image:
            image = np.array(image)
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        # input image density range [-1, 1]
        latents = self.vae.encode(image)['latent_dist'].mean
        latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        reference_image: PIL.Image.Image,
        source_subject_category: List[str],
        target_subject_category: List[str],
        latents: Optional[torch.FloatTensor] = None,
        source_prompt: Optional[str] = "",
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        neg_prompt: Optional[str] = "",
        prompt_strength: float = 1.0,
        prompt_reps: int = 20,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ):
        """
        Based on BlipDiffusionPipeline with just a few changes, the changes are below the line commented with "FreeCustom"
        """
        device = self._execution_device

        reference_image = self.image_processor.preprocess(
            reference_image, image_mean=self.config.mean, image_std=self.config.std, return_tensors="pt"
        )["pixel_values"]
        reference_image = reference_image.to(device)

        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(source_subject_category, str):
            source_subject_category = [source_subject_category]
        if isinstance(target_subject_category, str):
            target_subject_category = [target_subject_category]

        prompt = self._build_prompt(
            prompts=prompt,
            tgt_subjects=target_subject_category,
            prompt_strength=prompt_strength,
            prompt_reps=prompt_reps,
        )
        query_embeds = self.get_query_embeddings(reference_image, source_subject_category)
        text_embeddings_1 = self.encode_prompt(query_embeds, prompt, device)
        text_embeddings_2 = self.encode_prompt(query_embeds, source_prompt, device)
        text_embeddings = torch.cat([text_embeddings_1, text_embeddings_2], dim=0)
        
        batch_size = len(text_embeddings)
        
        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            max_length = self.text_encoder.text_model.config.max_position_embeddings

            uncond_input = self.tokenizer(
                [neg_prompt] * batch_size,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(
                input_ids=uncond_input.input_ids.to(device),
                ctx_embeddings=None,
            )[0]
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        scale_down_factor = 2 ** (len(self.unet.config.block_out_channels) - 1)
        latents = self.prepare_latents(
            batch_size=batch_size,
            num_channels=self.unet.config.in_channels,
            height=height // scale_down_factor,
            width=width // scale_down_factor,
            generator=generator,
            latents=latents,
            dtype=self.unet.dtype,
            device=device,
        )
        # FreeCustom
        latent_own_z_T, latents_ref_z_0 = latents[:1], latents[1:]
        
        # set timesteps
        extra_set_kwargs = {}
        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            # expand the latents if we are doing classifier free guidance
            do_classifier_free_guidance = guidance_scale > 1.0
            
            # FreeCustom
            latent_own, latents_ref = latents[:1], latents[1:]
            noise = randn_tensor(latents_ref_z_0.shape, generator=generator, device=device, dtype=latents_ref_z_0.dtype)
            latents_ref = self.scheduler.add_noise(latents_ref_z_0, noise, t.reshape(1,),)
            latents = torch.cat([latent_own, latents_ref], dim=0)
                
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

            noise_pred = self.unet(
                latent_model_input,
                timestep=t,
                encoder_hidden_states=text_embeddings,
                down_block_additional_residuals=None,
                mid_block_additional_residual=None,
            )["sample"]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(
                noise_pred,
                t,
                latents,
            )["prev_sample"]

        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
