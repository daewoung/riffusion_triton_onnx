import os

os.environ[
    "TRANSFORMERS_CACHE"
] = "/opt/tritonserver/model_repository/text_preprocess/hf_cache/snapshots/8f2e752c74e8316c6eb4fdaa6598a46ce1d88af5"

import json
import numpy as np
import torch
from transformers import CLIPTokenizer
import triton_python_backend_utils as pb_utils
from diffusers import DiffusionPipeline
import re
import torch.utils.dlpack
from torch.utils.dlpack import to_dlpack


class TritonPythonModel:
  def initialize(self, args):
    """
    change model
    """
    MODEL_NAME = "riffusion/riffusion-model-v1"
    pipeline = DiffusionPipeline.from_pretrained("/opt/tritonserver/model_repository/text_preprocess/hf_cache/snapshots/8f2e752c74e8316c6eb4fdaa6598a46ce1d88af5")
    #self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    self.tokenizer = pipeline.tokenizer
    self.device = 'cuda'
    self.use_reweighting = True
    self.torch_float_dtype = torch.float16 
    
    self.np_dtype_float = np.float16
    self.np_dtype_int = np.int16 

    
    self.re_attention = re.compile(
        r"""
    \\\(|
    \\\)|
    \\\[|
    \\]|
    \\\\|
    \\|
    \(|
    \[|
    :([+-]?[.\d]+)\)|
    \)|
    ]|
    [^\\()\[\]:]+|
    :
    """,
        re.X,
    )


  def parse_prompt_attention(self, text):
      """
      Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
      Accepted tokens are:
        (abc) - increases attention to abc by a multiplier of 1.1
        (abc:3.12) - increases attention to abc by a multiplier of 3.12
        [abc] - decreases attention to abc by a multiplier of 1.1
        \( - literal character '('
        \[ - literal character '['
        \) - literal character ')'
        \] - literal character ']'
        \\ - literal character '\'
        anything else - just text
      >>> parse_prompt_attention('normal text')
      [['normal text', 1.0]]
      >>> parse_prompt_attention('an (important) word')
      [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
      >>> parse_prompt_attention('(unbalanced')
      [['unbalanced', 1.1]]
      >>> parse_prompt_attention('\(literal\]')
      [['(literal]', 1.0]]
      >>> parse_prompt_attention('(unnecessary)(parens)')
      [['unnecessaryparens', 1.1]]
      >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
      [['a ', 1.0],
      ['house', 1.5730000000000004],
      [' ', 1.1],
      ['on', 1.0],
      [' a ', 1.1],
      ['hill', 0.55],
      [', sun, ', 1.1],
      ['sky', 1.4641000000000006],
      ['.', 1.1]]
      """

      res = []
      round_brackets = []
      square_brackets = []

      round_bracket_multiplier = 1.1
      square_bracket_multiplier = 1 / 1.1

      def multiply_range(start_position, multiplier):
          for p in range(start_position, len(res)):
              res[p][1] *= multiplier

      for m in self.re_attention.finditer(text):
          text = m.group(0)
          weight = m.group(1)

          if text.startswith("\\"):
              res.append([text[1:], 1.0])
          elif text == "(":
              round_brackets.append(len(res))
          elif text == "[":
              square_brackets.append(len(res))
          elif weight is not None and len(round_brackets) > 0:
              multiply_range(round_brackets.pop(), float(weight))
          elif text == ")" and len(round_brackets) > 0:
              multiply_range(round_brackets.pop(), round_bracket_multiplier)
          elif text == "]" and len(square_brackets) > 0:
              multiply_range(square_brackets.pop(), square_bracket_multiplier)
          else:
              res.append([text, 1.0])

      for pos in round_brackets:
          multiply_range(pos, round_bracket_multiplier)

      for pos in square_brackets:
          multiply_range(pos, square_bracket_multiplier)

      if len(res) == 0:
          res = [["", 1.0]]

      # merge runs of identical weights
      i = 0
      while i + 1 < len(res):
          if res[i][1] == res[i + 1][1]:
              res[i][0] += res[i + 1][0]
              res.pop(i + 1)
          else:
              i += 1

      return res
    
  def get_prompts_with_weights(self, prompt, max_length):
    r"""
    Tokenize a list of prompts and return its tokens with weights of each token.
    No padding, starting or ending token is included.
    """
    tokens = []
    weights = []
    truncated = False
    for text in prompt:
        texts_and_weights = self.parse_prompt_attention(text)
        text_token = []
        text_weight = []
        for word, weight in texts_and_weights:
            # tokenize and discard the starting and the ending token
            token = self.tokenizer(word).input_ids[1:-1]
            text_token += token
            # copy the weight by length of token
            text_weight += [weight] * len(token)
            # stop if the text is too long (longer than truncation limit)
            if len(text_token) > max_length:
                truncated = True
                break
        # truncate
        if len(text_token) > max_length:
            truncated = True
            text_token = text_token[:max_length]
            text_weight = text_weight[:max_length]
        tokens.append(text_token)
        weights.append(text_weight)

    return tokens, weights
  
  def pad_tokens_and_weights(
      self, tokens, weights, max_length, bos, eos, no_boseos_middle=False, chunk_length=77
  ):
      r"""
      Pad the tokens (with starting and ending tokens) and weights (with 1.0) to max_length.
      """
      max_embeddings_multiples = (max_length - 2) // (chunk_length - 2)
      weights_length = max_length if no_boseos_middle else max_embeddings_multiples * chunk_length
      for i in range(len(tokens)):
          tokens[i] = [bos] + tokens[i] + [eos] * (max_length - 1 - len(tokens[i]))
          if no_boseos_middle:
              weights[i] = [1.0] + weights[i] + [1.0] * (max_length - 1 - len(weights[i]))
          else:
              w = []
              if len(weights[i]) == 0:
                  w = [1.0] * weights_length
              else:
                  for j in range(max_embeddings_multiples):
                      w.append(1.0)  # weight for starting token in this chunk
                      w += weights[i][
                          j * (chunk_length - 2) : min(len(weights[i]), (j + 1) * (chunk_length - 2))
                      ]
                      w.append(1.0)  # weight for ending token in this chunk
                  w += [1.0] * (weights_length - len(w))
              weights[i] = w[:]

      return tokens, weights

  def execute(self, requests):
    responses = []
    for request in requests:
        use_reweighting = pb_utils.get_input_tensor_by_name(request, "use_reweighting")
        use_reweighting = use_reweighting.as_numpy()[0].astype(self.np_dtype_int)
        
        if int(use_reweighting) == 1:

            inp = pb_utils.get_input_tensor_by_name(request, "text_input")
            input_text = inp.as_numpy()[0].decode() 
            max_embeddings_multiples = 3
            max_length = (self.tokenizer.model_max_length -2) * max_embeddings_multiples + 2
            prompt_tokens, prompt_weights = self.get_prompts_with_weights([input_text], max_length -2)
            max_length = max([len(token) for token in prompt_tokens])
            
            max_embeddings_multiples = min(
            max_embeddings_multiples, (max_length - 1) // (self.tokenizer.model_max_length - 2) + 1)
            max_embeddings_multiples = max(1, max_embeddings_multiples)
            max_length = (self.tokenizer.model_max_length - 2) * max_embeddings_multiples + 2
            bos = self.tokenizer.bos_token_id
            eos = self.tokenizer.eos_token_id
            prompt_tokens, prompt_weights = self.pad_tokens_and_weights(prompt_tokens,
                    prompt_weights,
                    max_length,
                    bos,
                    eos,
                    no_boseos_middle=False,
                    chunk_length=self.tokenizer.model_max_length,
            )

            text_input = torch.tensor(prompt_tokens, dtype=torch.long)
            chunk_length=self.tokenizer.model_max_length
            
            max_embeddings_multiples = (text_input.shape[1] - 2) // (chunk_length - 2)
            prompt_weights = torch.tensor(prompt_weights, dtype=torch.float32)

            if max_embeddings_multiples > 1:
                text_embeddings = []
                for i in range(max_embeddings_multiples):
                    text_input_chunk = text_input[
                        :, i * (chunk_length - 2) : (i + 1) * (chunk_length - 2) + 2
                    ].clone()
                    text_input_chunk[:, 0] = text_input[0, 0]
                    text_input_chunk[:, -1] = text_input[0, -1]
                    input_ids_np = text_input_chunk.numpy().astype(np.int32)
                    input_ids_tensor = pb_utils.Tensor("input_ids", input_ids_np)
                    inference_request = pb_utils.InferenceRequest(model_name = "text_encoder", 
                                                                requested_output_names=["last_hidden_state__0"],
                                                                inputs = [input_ids_tensor],)
                    
                    inference_response = inference_request.exec()
            
                    if inference_response.has_error():
                        raise pb_utils.TritonModelException(inference_response.error().message())
                    else:
                    # text_emb = pb_utils.get_output_tensor_by_name(inference_response, "last_hidden_state__0").as_numpy()
                        # text_embeddings.append(torch.from_numpy(text_emb))
    
                        text_emb = pb_utils.get_output_tensor_by_name(inference_response, "last_hidden_state__0").to_dlpack()
                        text_embeddings.append(torch.from_dlpack(text_emb).to('cpu'))
            
            
                text_embeddings = torch.concat(text_embeddings, axis=1)
                previous_mean = text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
                text_embeddings *= prompt_weights.unsqueeze(-1)
                current_mean = text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
                text_embeddings *= (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)
                text_embeddings = text_embeddings.numpy()
                output = pb_utils.Tensor("output__0", text_embeddings.astype(self.np_dtype_float))
                
                inference_request = pb_utils.InferenceResponse(output_tensors=[output])
                responses.append(inference_request)    
        
            else:
                input_ids_np = text_input.numpy().astype(np.int32)
                input_ids_tensor = pb_utils.Tensor("input_ids", input_ids_np)
                inference_request = pb_utils.InferenceRequest(model_name = "text_encoder", 
                                                                requested_output_names=["last_hidden_state__0"],
                                                                inputs = [input_ids_tensor],)
                
                inference_response = inference_request.exec()
                
                if inference_response.has_error():
                    raise pb_utils.TritonModelException(inference_response.error().message())
                else:
                    text_emb = pb_utils.get_output_tensor_by_name(inference_response, "last_hidden_state__0").to_dlpack()
                    #text_emb = np.from_dlpack(text_emb)
                    #text_embeddings = torch.from_numpy(text_emb)
                    text_embeddings = torch.from_dlpack(text_emb).to('cpu')
                    previous_mean = text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
                    text_embeddings *= prompt_weights.unsqueeze(-1)
                    current_mean = text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
                    text_embeddings *= (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)
                    text_embeddings = text_embeddings.to(self.device)
                    
                    output = pb_utils.Tensor.from_dlpack("output__0", to_dlpack(text_embeddings))
                
                    inference_request = pb_utils.InferenceResponse(output_tensors=[output])
                    responses.append(inference_request)    
      
        elif int(use_reweighting) == 0:
            
            inp = pb_utils.get_input_tensor_by_name(request, "text_input")
            input_text = inp.as_numpy()[0].decode()
            input_ids = self.tokenizer([input_text], return_tensors="pt", padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True).input_ids
            input_ids_np = input_ids.numpy().astype(np.int32)
            input_ids_tensor = pb_utils.Tensor("input_ids", input_ids_np)
            
            inference_request = pb_utils.InferenceRequest(model_name = "text_encoder", 
                                                        requested_output_names=["last_hidden_state__0"],
                                                        inputs = [input_ids_tensor],)
            
            inference_response = inference_request.exec()
            if inference_response.has_error():
                raise pb_utils.TritonModelException(inference_response.error().message())
            else:
                text_emb = pb_utils.get_output_tensor_by_name(inference_response, "last_hidden_state__0").to_dlpack()
                #text_embeddings = text_embeddings.numpy().astype(np.float32)
 #               output = pb_utils.Tensor("output__0", text_embeddings.astype(np.float32))
                text_embeddings = torch.from_dlpack(text_emb).to(self.device)

                output = pb_utils.Tensor.from_dlpack("output__0", to_dlpack(text_embeddings))

                inference_request = pb_utils.InferenceResponse(output_tensors=[output])
                responses.append(inference_request)
    return responses 

  # def finalize(self):
  #       """`finalize` is called only once when the model is being unloaded.
  #       Implementing `finalize` function is optional. This function allows
  #       the model to perform any necessary clean ups before exit.
  #       """
  #       print("Cleaning up...")



