import torch

# def replace_model_weights(model_to_replace,model_with_weights):
#     state_dict_to_replace = model_to_replace.state_dict()
#     state_dict_with_weights = model_with_weights.state_dict()
#     # Filter common keys from both models
#     common_keys = set(state_dict_to_replace.keys()) & set(state_dict_with_weights.keys())
#     # Update the weights of the model_to_replace with the weights from model_with_weights
#     for key in common_keys:
#         state_dict_to_replace[key] = state_dict_with_weights[key]
#     # Load the updated state_dict back into model_to_replace
#     model_to_replace.load_state_dict(state_dict_to_replace)
#
#
# replace_model_weights(model_to_replace, model_with_weights)
# torch.save(model_to_replace.state_dict(), 'new_model_weights.pth')

# import copy
# url_or_filename = 'sevila_checkpoints/sevila_pretrained.pth'
# checkpoint = torch.load(url_or_filename, map_location="cpu")
# state_dict = checkpoint["model"]
#
# url_or_filename_new = '/scratch_xijun/code/Video/SeViLA/lavis/output/BLIP2/NextQA/QA/qvh_pretraining/checkpoint_best.pth'
# checkpoint_new = torch.load(url_or_filename_new, map_location="cpu")
# state_dict_new = checkpoint_new["model"]
#
# common_keys = set(state_dict.keys()) & set(state_dict_new.keys())
#
# key_test = None
# for key in common_keys:
#     key_test = key
#     state_dict[key] = copy.deepcopy(state_dict_new[key])
#     print(key)
#
# save_obj = {"model": state_dict}
# torch.save(save_obj, 'sevila_checkpoints/sevila_tt_pretrained.pth')


url_or_filename = '/prakhar/lamawaves/hub/checkpoints/instruct_blip_flanxl_trimmed.pth'
checkpoint = torch.load(url_or_filename, map_location="cpu")
state_dict = checkpoint["model"]

url_or_filename_loc = '/scratch_xijun/code/Video/SeViLA/lavis/output/BLIP2/NextQA/QA/qvh_pretraining_instruct/checkpoint_best.pth'
checkpoint_loc = torch.load(url_or_filename_loc, map_location="cpu")
state_dict_loc = checkpoint_loc["model"]

for key in state_dict_loc.keys():
    if 'loc' in key:
        state_dict[key] = state_dict_loc[key]
        print(key)

save_obj = {"model": state_dict}
torch.save(save_obj, 'sevila_checkpoints/instruct_blip_flanxl_trimmed_loc.pth')

#
# url_or_filename_new = 'sevila_checkpoints/sevila_tt_pretrained.pth'
# checkpoint_new = torch.load(url_or_filename_new, map_location="cpu")
# state_dict_new = checkpoint_new["model"]
#
# url_or_filename_tt = '/scratch_xijun/code/Video/SeViLA/lavis/output/BLIP2/NextQA/QA/qvh_pretraining/checkpoint_best.pth'
# checkpoint_tt = torch.load(url_or_filename_tt, map_location="cpu")
# state_dict_tt = checkpoint_tt["model"]
#
#
# print('######################')
# print(state_dict['Qformer_loc.bert.encoder.layer.5.attention.self.query.weight'])
# print('######################')
# print(state_dict_new['Qformer_loc.bert.encoder.layer.5.attention.self.query.weight'])
# print('######################')
# print(state_dict_tt['Qformer_loc.bert.encoder.layer.5.attention.self.query.weight'])
