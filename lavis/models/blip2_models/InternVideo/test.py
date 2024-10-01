def forward(self, samples):
    # print('-----------------')
    # print(samples["text_input"])
    # print(samples["text_output"])
    # print('-----------------')

    image = samples["video"]
    text_input = samples['loc_input']  # query + options + Prompt
    bs_answer = samples['qa_output']  # yes or no
    flat_answer = []
    for answer in bs_answer:
        answer = answer.split('_')
        for a in answer:
            flat_answer.append(a)

    b, t, c, w, h = image.shape
    image = image.reshape(-1, c, w, h)
    with self.maybe_autocast():
        image_embeds = self.ln_vision_loc(self.visual_encoder(image))
    _, n, _ = image_embeds.shape
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

    query_tokens = self.query_tokens_loc.expand(image_embeds.shape[0], -1, -1)

    text_Qformer = self.tokenizer(
        samples["loc_input"],
        padding='longest',
        truncation=True,
        max_length=self.max_txt_len,
        return_tensors="pt",
    ).to(image.device)
    text_Qformer_input_ids = torch.repeat_interleave(text_Qformer.input_ids, t, 0)
    text_Qformer_attention_mask = torch.repeat_interleave(text_Qformer.attention_mask, t, 0)
    query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
    Qformer_atts = torch.cat([query_atts, text_Qformer_attention_mask], dim=1)

    query_output = self.Qformer_loc.bert(
        text_Qformer_input_ids,
        attention_mask=Qformer_atts,
        query_embeds=query_tokens,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        return_dict=True,
    )

    inputs_t5 = self.t5_proj_loc(query_output.last_hidden_state[:, :query_tokens.size(1), :])
    atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

    fs_embeds, fs_atts = None, None
    if self.few_shot_prob > 0 and "few_shot_samples" in samples.keys():
        fs_embeds, fs_atts = self.prepare_few_shot_embeds(samples['few_shot_samples'])

    with self.maybe_autocast(dtype=torch.bfloat16):
        input_tokens = self.t5_tokenizer(
            samples["loc_input"],
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        input_ids = torch.repeat_interleave(input_tokens.input_ids, t, 0)
        input_attention_mask = torch.repeat_interleave(input_tokens.attention_mask, t, 0)

        output_tokens = self.t5_output_tokenizer(
            flat_answer,
            padding="longest",
            truncation=True,
            max_length=self.max_output_txt_len,
            return_tensors="pt",
        ).to(image.device)

        encoder_atts = torch.cat([atts_t5, input_attention_mask], dim=1)

        targets = output_tokens.input_ids.masked_fill(
            output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
        )

        inputs_embeds = self.t5_model.encoder.embed_tokens(input_ids)
        inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

        if fs_embeds is not None:
            inputs_embeds = torch.cat([fs_embeds, inputs_embeds], dim=1)
            encoder_atts = torch.cat([fs_atts, encoder_atts], dim=1)

        outputs = self.t5_model(
            inputs_embeds=inputs_embeds,
            attention_mask=encoder_atts,
            decoder_attention_mask=output_tokens.attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss

        return {"loss": loss}