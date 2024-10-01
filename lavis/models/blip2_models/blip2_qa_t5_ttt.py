"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from torch.cuda.amp import autocast as autocast
from transformers import T5TokenizerFast

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration
from lavis.models.topk import TopK_Selector

@registry.register_model("blip2_qa_t5_ttt")
class Blip2QAT5TTT(Blip2Base):
    """
    BLIP2 T5 model.
    Supported model types:
        - pretrain_flant5xl: pretrained model with FlanT5-XL
        - pretrain_flant5xxl: pretrained model with FlanT5-XXL
        - caption_coco_flant5xl: fintuned image captioning model with FlanT5-XL
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_t5", "pretrain_flant5xl")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_flant5xl": "configs/models/blip2/blip2_pretrain_flant5xl.yaml",
        "pretrain_flant5xxl": "configs/models/blip2/blip2_pretrain_flant5xxl.yaml",
        "caption_coco_flant5xl": "configs/models/blip2/blip2_caption_flant5xl.yaml",
    }

    def __init__(
        self,
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        t5_model="google/flan-t5-xl",
        prompt="",
        max_txt_len=32,
        apply_lemmatizer=False,
        freeze_qformer=True,
        freeze_query=True,
        freeze_proj=True,
        frame_num=4,
        freeze_decoder=True,
        task='qa'
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()

        self.task = task

        self.frame_num=frame_num
        
        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False         
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
                 
        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model)
        # add span tokens
        span_tokens = ['<span{}>'.format(str(i)) for i in range(15)]
        self.frame_ids = [ '<extra_id_{}>'.format(i) for i in range(frame_num)]
        self.frame_prefix = ['Frame: ']
        self.t5_tokenizer.add_tokens(span_tokens, special_tokens=True)
        
        t5_config = T5Config.from_pretrained(t5_model)
        t5_config.dense_act_fn = "gelu"
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            t5_model, config=t5_config
        )
        #
        for name, param in self.t5_model.named_parameters():
            if ('shared' in name) and (self.task=='mr'):
                logging.info("finetune T5 embedding layer")
                continue
            elif ('lm_head' in name) and (self.task=='mr'):
                logging.info("finetune T5 lm_head layer")
                continue
            elif ('decoder' in name) and (not freeze_decoder):
                logging.info("finetune {}".format(name))
                continue
            else:
                param.requires_grad = False
                param.data = param.data.bfloat16()

        self.t5_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.t5_model.config.hidden_size
        )
        
        if self.task in ['mr', 'locqa']: 
            self.temp_proj = nn.Linear(
                self.t5_model.config.hidden_size, 1408
            )
            print('t5 dim:', self.t5_model.config.hidden_size)
            
        self.max_txt_len = 65 #max_txt_len
        self.prompt = prompt
        
        self.answer_id = [71, 272, 205, 309, 262] # A B C D E

        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None

        #self.vqproj = nn.Sequential(nn.Dropout(0.1), nn.Linear(2048, 2048))

        self.prompt_pool = nn.Parameter(torch.zeros(1, 2048))
        trunc_normal_(self.prompt_pool, std=0.02)


    def forward(self, samples):
        image = samples["video"]
        #print('image1', image.shape) # b t c w n
        b, t, c, w, h = image.shape

        # tt
        text = self.predict_answers(samples)
        for i in range(len(samples['qa_input'])):
            samples['qa_input'][i] = 'Question: a photo of? Answer: ' + ', '.join(text[self.frame_num*i:self.frame_num*(i+1)]) +'. ' + samples['qa_input'][i]
        # tt

        # image = image.reshape(-1, c, w, h)
        # image_embeds = self.ln_vision(self.visual_encoder(image)) # bt, n, c
        # _, n, _ = image_embeds.shape
        #
        # #print('image_embeds.shape:', image_embeds.shape)
        # #print('self.visual_encoder.num_features', self.visual_encoder.num_features)
        # if self.task in ['qa', 'mr', 'locqa']:
        #     image_embeds = image_embeds.reshape(b, -1, image_embeds.shape[-1])
        #
        # image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
        #         image.device
        #     )
        # # b tn c
        #
        # if self.task in ['mr', 'locqa']:
        #     with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        #
        #         temp_pos = self.t5_tokenizer(
        #                 self.frame_ids,
        #                 padding="longest",
        #                 truncation=True,
        #                 max_length=self.max_txt_len,
        #                 return_tensors="pt",
        #             ).to(image.device)
        #         temp_pos = self.t5_model.encoder.embed_tokens(temp_pos.input_ids[:, 0]) # t, d
        #         temp_pos = torch.repeat_interleave(temp_pos, n, dim=0) # tn d
        #         temp_pos = temp_pos.unsqueeze(0)
        #         temp_pos = torch.repeat_interleave(temp_pos, b, dim=0)
        #
        #     temp_pos = temp_pos.to(torch.float32)
        #     temp_pos = self.temp_proj(temp_pos)
        #     image_embeds += temp_pos
        
        # query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        # query_output = self.Qformer.bert(
        #         query_embeds=query_tokens,
        #         encoder_hidden_states=image_embeds,
        #         encoder_attention_mask=image_atts,
        #         return_dict=True,
        #     )
        #
        # inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        # atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)
        
        if self.task in ['qa', 'frameqa']:
            text_input = samples['qa_input']
            text_output = samples['qa_output']
        elif self.task in ['mr']:
            text_input = samples['mr_input']
            text_output = samples['mr_output']
        elif self.task in ['locqa']:
            text_input = samples['locqa_input']
            text_output = samples['sequence_label']
              
        # print('text_input', text_input)
        # print('text_output', text_output)
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            frame_prefix = self.t5_tokenizer(
            self.frame_prefix,padding="longest", add_special_tokens=False,
            truncation=True, max_length=self.max_txt_len,return_tensors="pt",
            ).to(image.device) # 
            if self.task in ['frameqa']:
                frame_prefix_id = torch.repeat_interleave(frame_prefix.input_ids, b*t, 0)
                frame_prefix_mask = torch.repeat_interleave(frame_prefix.attention_mask, b*t, 0)
            else:
                frame_prefix_id = torch.repeat_interleave(frame_prefix.input_ids, b, 0)
                frame_prefix_mask = torch.repeat_interleave(frame_prefix.attention_mask, b, 0)
                

            input_tokens = self.t5_tokenizer(
                    text_input,
                    padding="longest",
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(image.device)

            if self.task in ['qa', 'mr', 'frameqa']:
                output_tokens = self.t5_tokenizer(
                        text_output,
                        padding="longest",
                        truncation=True,
                        max_length=self.max_txt_len,
                        return_tensors="pt",
                    ).to(image.device)
                output_tokens_id = output_tokens.input_ids
                output_tokens_mask = output_tokens.attention_mask
                
            elif self.task in ['locqa']:
                #print('text_output', text_output)
                output_tokens_id = text_output.to(image.device)
                #print('output_tokens_id', output_tokens_id.shape)
                output_tokens_mask = torch.ones_like(output_tokens_id).to(image.device)
                
            #encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

            targets = output_tokens_id.masked_fill(
                    output_tokens_id == self.t5_tokenizer.pad_token_id, -100
                )
            #print('target', targets)
            
            # if self.task == 'frameqa':
            #     output_tokens_mask = torch.repeat_interleave(output_tokens_mask, t, dim=0)
            #     targets = torch.repeat_interleave(targets, t, dim=0)
            #     input_attention_mask = torch.repeat_interleave(input_tokens.attention_mask, t, 0)
            #     encoder_atts = torch.cat([frame_prefix_mask, atts_t5, input_attention_mask], dim=1)
            #     input_ids = torch.repeat_interleave(input_tokens.input_ids, t, 0)
            #     inputs_embeds = self.t5_model.encoder.embed_tokens(input_ids)
            #     frame_predix_embed = self.t5_model.encoder.embed_tokens(frame_prefix_id)
            #     inputs_embeds = torch.cat([frame_predix_embed, inputs_t5, inputs_embeds], dim=1)
            # else:
            #     encoder_atts = torch.cat([frame_prefix_mask, atts_t5, input_tokens.attention_mask], dim=1)
            #     frame_predix_embed = self.t5_model.encoder.embed_tokens(frame_prefix_id)
            #     inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            #
            #     inputs_embeds = torch.cat([frame_predix_embed, inputs_t5, inputs_embeds], dim=1)
            #
            # outputs = self.t5_model(
            #         inputs_embeds=inputs_embeds,
            #         attention_mask=encoder_atts,
            #         decoder_attention_mask=output_tokens_mask,
            #         return_dict=True,
            #         labels=targets,
            #     )

            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds_att = inputs_embeds.mean((1,2))
            inputs_embeds_prompt = torch.matmul(inputs_embeds_att.unsqueeze(dim=1), self.prompt_pool)
            inputs_embeds_prompt = inputs_embeds_prompt.unsqueeze(dim=1)
            inputs_embeds = torch.cat([inputs_embeds, inputs_embeds_prompt], dim=1)
            encoder_atts = torch.cat([input_tokens.attention_mask, torch.ones(inputs_embeds_prompt.size()[:-1], dtype=torch.long).to(image.device)], dim=1)

            outputs = self.t5_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=encoder_atts,
                    decoder_attention_mask=output_tokens_mask,
                    return_dict=True,
                    labels=targets,
                )
            loss = outputs.loss

            return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        out = {}
        image, qid = samples["video"], samples['question_id'] 
        b, t, c, w, h = image.shape

        # tt
        text = self.predict_answers(samples)
        for i in range(len(samples['qa_input'])):
            samples['qa_input'][i] = 'Question: a photo of? Answer: ' + ', '.join(text[self.frame_num*i:self.frame_num*(i+1)]) +'. ' + samples['qa_input'][i]
        # tt

        image = image.reshape(-1, c, w, h) # bt c w h

        if self.task in ['qa','frameqa']:
            prompt = samples['qa_input']
            answer = samples['qa_output']
        elif self.task in ['mr']:
            prompt = samples['mr_input']
            answer = samples['mr_output']
        elif self.task in ['locqa']:
            prompt = samples['locqa_input']
            answer = samples['qa_output']

        assert len(prompt) == b, "The number of prompts must be equal to the batch size."
        #
        # with torch.cuda.amp.autocast(enabled=(self.device != torch.device("cpu"))):
        #     image_embeds = self.ln_vision(self.visual_encoder(image))
        #
        # _, n, _ = image_embeds.shape
        # if self.task in ['qa', 'mr', 'locqa']:
        #     image_embeds = image_embeds.reshape(b, -1, image_embeds.shape[-1])
        #
        # image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
        #     image.device
        # )
        #
        # if self.task in ['mr', 'locqa']:
        #     with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        #
        #         temp_pos = self.t5_tokenizer(
        #                 self.frame_ids,
        #                 padding="longest",
        #                 truncation=True,
        #                 max_length=self.max_txt_len,
        #                 return_tensors="pt",
        #             ).to(image.device)
        #         temp_pos = self.t5_model.encoder.embed_tokens(temp_pos.input_ids[:, 0]) # t, d
        #         temp_pos = torch.repeat_interleave(temp_pos, n, dim=0) # tn d
        #         temp_pos = temp_pos.unsqueeze(0)
        #         temp_pos = torch.repeat_interleave(temp_pos, b, dim=0)
        #     # print('temp1', temp_pos.shape, temp_pos.dtype) # 4 8224 2048
        #     # print('img', image_embeds.shape) #
        #     temp_pos = temp_pos.to(torch.float32)
        #     temp_pos = self.temp_proj(temp_pos)
        #     image_embeds += temp_pos
        #
        # query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        # query_output = self.Qformer.bert(
        #     query_embeds=query_tokens,
        #     encoder_hidden_states=image_embeds,
        #     encoder_attention_mask=image_atts,
        #     return_dict=True,
        # )
        #
        # inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        # atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        input_tokens = self.t5_tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).to(image.device)
        
        if self.task in ['qa', 'frameqa']:
            output_tokens = self.t5_tokenizer(
                        samples["qa_output"],
                        padding="longest",
                        truncation=True,
                        max_length=self.max_txt_len,
                        return_tensors="pt",
                    ).to(image.device)

            targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
            )
            output_tokens_att_mask = output_tokens.attention_mask 
             
            if self.task == 'frameqa':
                output_tokens_att_mask = torch.repeat_interleave(output_tokens_att_mask, t, dim=0)
                targets = torch.repeat_interleave(targets, t, dim=0)
        
        # if self.task == 'frameqa':
        #     input_attention_mask = torch.repeat_interleave(input_tokens.attention_mask, t, 0)
        #     encoder_atts = torch.cat([atts_t5, input_attention_mask], dim=1)
        # else:
        #     encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)
        
        # A:71 B:272 C:205 D:309 E:262

        device_type = "cuda" if "cuda" in str(self.device) else "cpu"
        with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
            
            # if self.task == 'frameqa':
            #     input_ids = torch.repeat_interleave(input_tokens.input_ids, t, 0)
            #     inputs_embeds = self.t5_model.encoder.embed_tokens(input_ids)
            # else:
            #     inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            #
            # inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)


            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds_att = inputs_embeds.mean((1, 2))
            inputs_embeds_prompt = torch.matmul(inputs_embeds_att.unsqueeze(dim=1), self.prompt_pool)
            inputs_embeds_prompt = inputs_embeds_prompt.unsqueeze(dim=1)
            inputs_embeds = torch.cat([inputs_embeds, inputs_embeds_prompt], dim=1)
            encoder_atts = torch.cat([input_tokens.attention_mask,
                                      torch.ones(inputs_embeds_prompt.size()[:-1], dtype=torch.long).to(image.device)],
                                     dim=1)
            
            # if self.task in ['qa', 'frameqa']:
            #     outputs_embed = self.t5_model(
            #             inputs_embeds=inputs_embeds,
            #             attention_mask=encoder_atts,
            #             decoder_attention_mask=output_tokens_att_mask,
            #             return_dict=True,
            #             labels=targets,
            #         )
            #     #print('answer', answer)
            #     pred_logits = outputs_embed.logits
            #     pred_logits = pred_logits[:, 1, self.answer_id]
            #     pred_logits = pred_logits.softmax(-1)
            
            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=1,
                max_new_tokens=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
                return_dict_in_generate=True, output_hidden_states=True, output_scores=True)
            
            pred_logits_qa = outputs.scores[1] #outputs_embed_qa.logits.detach()
            pred_logits_qa = pred_logits_qa[:, self.answer_id] # b, 5
            #print('pred_logits_qa', pred_logits_qa.shape)
            pred_ans = torch.argmax(pred_logits_qa, dim=-1).cpu().tolist()
            output_text = pred_ans

            # if self.task in ['qa', 'frameqa', 'locqa']:
            #     output_text = self.t5_tokenizer.batch_decode(
            #         outputs, skip_special_tokens=True
            #     )
            #     #output_text = pred_ans
            # else:
            #     output_text = self.t5_tokenizer.batch_decode(
            #         outputs, skip_special_tokens=False
            #     )
                
        out['answer'] = answer
        out['qid'] = qid
        out['output_text'] = output_text
        
        if self.task in ['qa', 'frameqa']:
            # out['pred_logits'] = pred_logits.cpu().tolist() 
            
            if self.task in ['frameqa']:
                out['temp_idx'] = [ j for i in range(b) for j in range(t)]
                out['answer'] = [a for a in answer for i in range(t)]
                out['qid'] = [q for q in qid for i in range(t)]
            
        return out

    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=-1,
        **kwargs
    ):
        image = samples["video"]
        # print('image1', image.shape) # b t c h, w
        b, t, c, h, w = image.shape

        image = image.reshape(-1, c, h, w)
        with torch.cuda.amp.autocast(enabled=(self.device != torch.device("cpu"))):
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        # if isinstance(samples["text_input"], str):
        #     samples["text_input"] = [samples["text_input"]]
        # if prompt:
        #     text_input = [prompt.format(question) for question in samples["text_input"]]
        # else:
        #     text_input = samples["text_input"]

        text_input = ["a photo of" for _ in range(len(samples["qa_input"])*self.frame_num)]

        input_tokens = self.t5_tokenizer(
            text_input, padding="longest", return_tensors="pt"
        ).to(image.device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        device_type = "cuda" if "cuda" in str(self.device) else "cpu"
        with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=min_len,
                length_penalty=length_penalty,
            )
            output_text = self.t5_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

        if self._apply_lemmatizer:
            output_text = self._lemmatize(output_text)

        return output_text

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        t5_model = cfg.get("t5_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)
        
        freeze_qformer= cfg.get("freeze_qformer", True)
        freeze_query=cfg.get("freeze_query", True)
        freeze_proj=cfg.get("freeze_proj", True)
        freeze_decoder=cfg.get("freeze_decoder", True)
        frame_num=cfg.get("frame_num", 4)
        task = cfg.get("task", 'qa')

        print("frame_num:", frame_num)

        model = cls(
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            t5_model=t5_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer,
            freeze_qformer=freeze_qformer,
            freeze_query=freeze_query,
            freeze_proj=freeze_proj,
            frame_num=frame_num,
            freeze_decoder=freeze_decoder,
            task=task
        )
        model.load_checkpoint_from_config(cfg)

        return model