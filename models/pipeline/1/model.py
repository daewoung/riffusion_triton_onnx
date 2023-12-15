import os

os.environ[
    "TRANSFORMERS_CACHE"
] = "/opt/tritonserver/model_repository/text_preprocess/hf_cache/snapshots/8f2e752c74e8316c6eb4fdaa6598a46ce1d88af5"

import json
import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from diffusers import DiffusionPipeline
from torch.utils.dlpack import to_dlpack
import inspect
import logging
import time
from PIL import Image

class TritonPythonModel:
  def initialize(self, args):
    """
    change model
    """
    MODEL_NAME = "riffusion/riffusion-model-v1"
    pipeline = DiffusionPipeline.from_pretrained("/opt/tritonserver/model_repository/text_preprocess/hf_cache/snapshots/8f2e752c74e8316c6eb4fdaa6598a46ce1d88af5")
    
    self.scheduler = pipeline.scheduler
    
    self.torch_float_dtype = torch.float32 
    
    self.np_dtype_float = np.float32
    self.np_dtype_int = np.int32
    self.eta = 0.0
    self.device = 'cuda'
  def slerp(self,
    t: float, v0: torch.Tensor, v1: torch.Tensor, dot_threshold: float = 0.9995
  ) -> torch.Tensor:
    """
    Helper function to spherically interpolate two arrays v1 v2.
    """
    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > dot_threshold:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2
    
  def get_text_emb(self, text, use_reweighting):
    if not isinstance(text, str):
      text = str(text)
    text = np.array([text], dtype=object)
    text = pb_utils.Tensor("text_input", text)
    
    if use_reweighting == 1:
      reweight = np.array([1], dtype=self.np_dtype_int)
      reweight = pb_utils.Tensor("use_reweighting", reweight)

    elif use_reweighting == 0:
      reweight = np.array([0], dtype=self.np_dtype_int)
      reweight = pb_utils.Tensor("use_reweighting", reweight)

    emb_text_infer = pb_utils.InferenceRequest(model_name = "text_preprocess",  
                                            requested_output_names=["output__0"],
                                            inputs = [text, reweight])
    emb_text_infer = emb_text_infer.exec()
    
    if emb_text_infer.has_error():
        raise pb_utils.TritonModelException(emb_text_infer.error().message())
    
    emb_text = pb_utils.get_output_tensor_by_name(emb_text_infer, "output__0").to_dlpack()
    emb_text = torch.from_dlpack(emb_text)

    return emb_text
  
  def preprocess_image(self, image: Image.Image) -> torch.Tensor:
    """
    Preprocess an image for the model.
    """
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.LANCZOS)

    image_np = np.array(image).astype(np.float32) / 255.0
    image_np = image_np[None].transpose(0, 3, 1, 2)

    image_torch = torch.from_numpy(image_np)

    return 2.0 * image_torch - 1.0
  
  def get_init_image(self, img_pth):
    img = Image.open(img_pth).convert("RGB")
    img = self.preprocess_image(img)
    img = img.numpy().astype(self.np_dtype_float)
    
    input_img_tensor = pb_utils.Tensor("encoder_input__0", img)
    
    inference_request = pb_utils.InferenceRequest(model_name = "vae_encoder",  
                                                  requested_output_names=["sampled_tensor__0"],
                                                  inputs = [input_img_tensor],)
    inference_response = inference_request.exec()

    if inference_response.has_error():
      raise pb_utils.TritonModelException(inference_response.error().message())
    
    sampled_tensor = pb_utils.get_output_tensor_by_name(inference_response, "sampled_tensor__0").to_dlpack()
    init_latents = torch.from_dlpack(sampled_tensor)    
      
    return init_latents
  
  def get_input_tensor(self, request):
    start_guidance = pb_utils.get_input_tensor_by_name(request, "start_guidance")
    start_guidance = start_guidance.as_numpy()[0].astype(self.np_dtype_float)
    
    end_guidance = pb_utils.get_input_tensor_by_name(request, "end_guidance")
    end_guidance = end_guidance.as_numpy()[0].astype(self.np_dtype_float)
    
    alpha = pb_utils.get_input_tensor_by_name(request, "alpha")
    alpha = alpha.as_numpy()[0].astype(self.np_dtype_float)
    
    start_text_seed = pb_utils.get_input_tensor_by_name(request, "start_text_seed")
    start_text_seed = start_text_seed.as_numpy()[0].astype(self.np_dtype_int)
    end_text_seed = pb_utils.get_input_tensor_by_name(request, "end_text_seed")
    end_text_seed = end_text_seed.as_numpy()[0].astype(self.np_dtype_int)

    img_pth = pb_utils.get_input_tensor_by_name(request, "img_path")
    img_pth = img_pth.as_numpy()[0].decode() 
    
    start_text = pb_utils.get_input_tensor_by_name(request, "start_text")
    start_text = start_text.as_numpy()[0].decode() 
    
    end_text = pb_utils.get_input_tensor_by_name(request, "end_text")
    end_text = end_text.as_numpy()[0].decode() 
    
    num_inference_steps = pb_utils.get_input_tensor_by_name(request, "num_inference_steps")
    num_inference_steps = num_inference_steps.as_numpy()[0].astype(self.np_dtype_int)

    negative_prompt = pb_utils.get_input_tensor_by_name(request, "negative_prompt")
    negative_prompt = negative_prompt.as_numpy()[0].decode() 
    
    width = pb_utils.get_input_tensor_by_name(request, "width")
    width = width.as_numpy()[0].astype(self.np_dtype_int)
    
    num_images_per_prompt = pb_utils.get_input_tensor_by_name(request, "num_images_per_prompt")
    num_images_per_prompt = num_images_per_prompt.as_numpy()[0].astype(self.np_dtype_int)
    
    return start_guidance, end_guidance, alpha, start_text_seed, end_text_seed, img_pth, start_text, end_text, num_inference_steps, negative_prompt, width, num_images_per_prompt
  
  def run_unet(self, latent, t, text_embeddings):
    batch_size = latent.shape[0]
    #latent = latent.numpy().astype(np.float32)
    #t = np.array([[t]] * batch_size, dtype=np.float32)
    t = t.expand(batch_size, 1).type(self.torch_float_dtype)

    latent = pb_utils.Tensor.from_dlpack("sample__0", to_dlpack(latent))
    t = pb_utils.Tensor.from_dlpack("timesteps__0", to_dlpack(t))
    text_embeddings = pb_utils.Tensor.from_dlpack("encoder_hidden_states__0", to_dlpack(text_embeddings))

    inference_request = pb_utils.InferenceRequest(model_name = "unet",  
                                                  requested_output_names=["output__0"],
                                                  inputs = [latent, t, text_embeddings])
    inference_response = inference_request.exec()

    if inference_response.has_error():
      raise pb_utils.TritonModelException(inference_response.error().message())

    output = pb_utils.get_output_tensor_by_name(inference_response, "output__0").to_dlpack()
    output = torch.from_dlpack(output)

    return output
  
  def vae_decode(self, latents):
    latents = pb_utils.Tensor.from_dlpack("latent_sample__0", to_dlpack(latents))
    
    inference_request = pb_utils.InferenceRequest(model_name = "vae_decoder",  
                                                  requested_output_names=["sample__0"],
                                                  inputs = [latents],)
    inference_response = inference_request.exec()

    if inference_response.has_error():
      raise pb_utils.TritonModelException(inference_response.error().message())

    output = pb_utils.get_output_tensor_by_name(inference_response, "sample__0").to_dlpack()
    output = torch.from_dlpack(output)

    return output
  def execute(self, requests):
    responses = []
    # 1번째 텍스트 1개만 들어오는 경우
    # 2번째 - 텍스트 2개가 다 들어오는 경우 
    # + 경우 이미지 컨텍스트가 들어오는 경우 
    for request in requests:
      total_start_time = time.time()

      start_time = time.time()
      start_guidance, end_guidance, alpha, start_text_seed, end_text_seed, \
      img_pth, start_text, end_text, num_inference_steps, negative_prompt, width, num_images_per_prompt = self.get_input_tensor(request)
      end_time = time.time()
      get_input_tensor_time = end_time - start_time
      
      if start_text == end_text:
        start_time = time.time()
        start_text = self.get_text_emb(start_text, 1)
        end_time = time.time()

        get_start_text_time = end_time - start_time

        end_text = start_text
        text_embeddings = start_text
        generator_start = torch.Generator(start_text.device).manual_seed(int(start_text_seed))
        guidance_scale = start_guidance
        print('start_text == end_text')
        print('alpha', alpha, 'start_guidance', start_guidance, 'end_guidance', end_guidance, 'guidance_scale', guidance_scale, 'start_text', start_text, 'end_text', end_text, 'text_embeddings', text_embeddings)
        print(start_text.shape)

      else: 
        start_time = time.time()
        start_text = self.get_text_emb(start_text, 1)
        end_time = time.time()
        get_start_text_time = end_time - start_time
        start_time = time.time()
        end_text = self.get_text_emb(end_text, 1)
        end_time = time.time()
        get_end_text_time = end_time - start_time

        text_embeddings = start_text + alpha * (end_text - start_text)
        generator_start = torch.Generator(start_text.device).manual_seed(int(start_text_seed))
        generator_end = torch.Generator(end_text.device).manual_seed(int(end_text_seed))
        guidance_scale = start_guidance * (1.0 - alpha) + end_guidance * alpha
        print('start_text != end_text')
        print('alpha', alpha, 'start_guidance', start_guidance, 'end_guidance', end_guidance, 'guidance_scale', guidance_scale, 'start_text', start_text, 'end_text', end_text, 'text_embeddings', text_embeddings)

      start_time = time.time()
      uncond_embeddings = self.get_text_emb(negative_prompt, 0)
      end_time = time.time()
      get_uncod_text_time = end_time - start_time
      
      start_time = time.time()
      

      
      self.scheduler.set_timesteps(int(num_inference_steps))
      strength_a = 0.8 
      strength_b = 0.8
      
      batch_size = text_embeddings.shape[0]
      
      uncond_embeddings = uncond_embeddings.repeat_interleave(batch_size * num_images_per_prompt, dim=0)
      bs_embed, seq_len, _ = text_embeddings.shape
      text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
      text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

      

      text_embeddings = torch.cat([uncond_embeddings, text_embeddings], dim=0)
      text_embeddings = torch.tensor(text_embeddings, dtype = torch.float32, device = self.device)


    
      if img_pth == "None":
        img_pth = None
        latents_dtype = text_embeddings.dtype
        print(latents_dtype)
        do_classifier_free_guidance = guidance_scale > 1.0

        batch_size = 1 
        latents_shape = (
          batch_size * num_images_per_prompt,
          4,
          64,
          width // 8          
        )

        timesteps_tensor = self.scheduler.timesteps.to(self.device)
        
        latents = torch.randn(
                latents_shape, generator=generator_start, device=self.device, dtype=latents_dtype
            )
        latents = latents * self.scheduler.init_noise_sigma
        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = self.eta
        
        for i, t in enumerate(timesteps_tensor):
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t
            )
            noise_pred = self.run_unet(latent_model_input, t, text_embeddings)
            
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

      
      else:
        latents_dtype = text_embeddings.dtype
        
        strength = (1 - alpha) * strength_a + alpha * strength_b

        offset = self.scheduler.config.get("steps_offset", 0)
        init_timestep = int(num_inference_steps * strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)

        timesteps = self.scheduler.timesteps[-init_timestep]
        timesteps = torch.tensor(
            [timesteps] * batch_size * num_images_per_prompt, device=text_embeddings.device
        )

        noise_a = torch.randn(
              (batch_size, 4, 64, 64), generator=generator_start, device=text_embeddings.device, dtype=latents_dtype
          )
        noise_b = torch.randn(
            (batch_size, 4, 64, 64), generator=generator_end, device=text_embeddings.device, dtype=latents_dtype
        )
        
        noise = self.slerp(alpha, noise_a, noise_b)

        do_classifier_free_guidance = guidance_scale > 1.0
        end_time = time.time()
        get_utils_time = end_time - start_time
       # img_seed = pb_utils.get_input_tensor_by_name(request, "img_path_seed")
      #img_seed = img_seed.as_numpy()[0].astype(np.int32)
        start_time = time.time()
        init_latents = self.get_init_image(img_pth)
        end_time = time.time()
        get_init_image_time = end_time - start_time
        init_latents_orig = init_latents
        init_latents = self.scheduler.add_noise(init_latents, noise, timesteps)

        latents = init_latents.clone()
        
        t_start = max(num_inference_steps - init_timestep + offset, 0)
        timesteps = self.scheduler.timesteps[t_start:].to(latents.device)
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
          extra_step_kwargs["eta"] = self.eta 
        
        start_time = time.time()

        for i, t in enumerate(timesteps):
        
          lm1 = time.time()
          latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
          latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
          
          lm2 = time.time()
          
          lm3 = time.time()
          noise_pred = self.run_unet(latent_model_input, t, text_embeddings)
          lm4 = time.time()
           
          lm5 = time.time()            
          if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
          lm6 = time.time()
          
          lm7 = time.time()
          latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
          lm8 = time.time()
          unet1 = lm2 - lm1
          unet2 = lm4 - lm3
          unet3 = lm6 - lm5
          unet4 = lm8 - lm7

        end_time = time.time()
        run_unet_time = end_time - start_time

      start_time = time.time()
      latents = 1.0 / 0.18215 * latents
      image = self.vae_decode(latents)
      end_time = time.time()
      get_vae_decode_time = end_time - start_time
      image = (image / 2 + 0.5).clamp(0, 1)
      image = image.cpu().permute(0, 2, 3, 1).numpy()
      
      latents = latents.cpu().numpy()
      
      output = pb_utils.Tensor("output__0", image.astype(np.float32))
      inference_request = pb_utils.InferenceResponse(output_tensors=[output])
      responses.append(inference_request)
      total_end_time = time.time()
      inference_time = total_end_time - total_start_time
      #'uent_time:', unet2,
      print('total_time :', inference_time) #'vae_decode_time', get_vae_decode_time, 'text_encoder_time', get_start_text_time, "vae_encoder_time", get_init_image_time)
#      print('vae_decode:', get_vae_decode_time, 'vae_encode:', get_init_image_time, 'text_encoder_time', get_start_text_time,'uent_time:', unet2)
    return responses  