from lavis.models import load_model_and_preprocess
## tt
self.tt_model, _, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xxl",
                                                is_eval=True, device=torch.device("cuda"))
## tt


# tt
for i in range(b):
    for j in range(t):
        with torch.no_grad():
            tt = self.tt_model.generate({"image": image[i, j:j+1, :, :, :], "prompt": "a photo of"})
        tt = '' + tt
        samples['qa_input'][i] += tt
# tt