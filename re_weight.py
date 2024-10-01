import torch

#STAR 32 to 4
# 4 frames model
url_or_filename = '/scratch_xijun/code/Video/SeViLA/lavis/output/BLVQA/STAR/QA/blvqa_4f_2e-5_1000wu/checkpoint_best.pth'
checkpoint = torch.load(url_or_filename, map_location="cpu")
state_dict = checkpoint["model"]

# 8-32 frames model
url_or_filename_T = '/scratch_xijun/code/Video/SeViLA/lavis/output/BLVQA/STAR/QA/blvqa_32f/checkpoint_best.pth'
checkpoint_T = torch.load(url_or_filename_T, map_location="cpu")
state_dict_T = checkpoint_T["model"]

new_state_dict = state_dict.copy()


for key in state_dict.keys():
    if 'Qformer' in key:
        new_key = key.replace('Qformer', 'Qformer_T')
        new_state_dict[new_key] = state_dict_T[key]
        print(key)
    if 'query_tokens' in key:
        new_key = key.replace('query_tokens', 'query_tokens_T')
        new_state_dict[new_key] = state_dict_T[key]
        print(key)
    if 't5_proj' in key:
        new_key = key.replace('t5_proj', 't5_proj_T')
        new_state_dict[new_key] = state_dict_T[key]
        print(key)

save_obj = {"model": new_state_dict}
torch.save(save_obj, 'vila_checkpoints/star_vlap_blip_flanxl_trimmed_32t4T.pth')




# #NEXTQA
# url_or_filename = '/scratch_xijun/code/Video/SeViLA/lavis/output/BLIP2/NextQA/QA/blip2_nextqa_instruct_32f/checkpoint_best.pth'
# checkpoint = torch.load(url_or_filename, map_location="cpu")
# state_dict = checkpoint["model"]
#
# new_state_dict = state_dict.copy()
#
#
# for key in state_dict.keys():
#     if 'Qformer' in key:
#         new_key = key.replace('Qformer', 'Qformer_T')
#         new_state_dict[new_key] = state_dict[key]
#         print(key)
#     if 'query_tokens' in key:
#         new_key = key.replace('query_tokens', 'query_tokens_T')
#         new_state_dict[new_key] = state_dict[key]
#         print(key)
#     if 't5_proj' in key:
#         new_key = key.replace('t5_proj', 't5_proj_T')
#         new_state_dict[new_key] = state_dict[key]
#         print(key)
#
# save_obj = {"model": new_state_dict}
# torch.save(save_obj, 'sevila_checkpoints/nextqa_blvqa_blip_flanxl_trimmed_T.pth')


# url_or_filename = '/scratch_xijun/code/Video/SeViLA/lavis/output/BLIP2/NextQA/QA/blip2_nextqa_instruct_2e-5/checkpoint_best.pth'
# checkpoint = torch.load(url_or_filename, map_location="cpu")
# state_dict = checkpoint["model"]
#
# url_or_filename_new = 'sevila_checkpoints/instruct_blip_flanxl_trimmed_T.pth'
# checkpoint_new = torch.load(url_or_filename_new, map_location="cpu")
# state_dict_new = checkpoint_new["model"]
#
# for key in state_dict.keys():
#     if 'Qformer' in key:
#         new_key = key.replace('Qformer', 'Qformer_T')
#         if key in set(state_dict_new.keys()) and new_key in set(state_dict_new.keys()):
#             continue
#         else:
#             print('Qformer:', key)
#
#     if 'query_tokens' in key:
#         new_key = key.replace('query_tokens', 'query_tokens_T')
#         if key in set(state_dict_new.keys()) and new_key in set(state_dict_new.keys()):
#             continue
#         else:
#             print('query_tokens:', key)
#     if 't5_proj' in key:
#         new_key = key.replace('t5_proj', 't5_proj_T')
#         if key in set(state_dict_new.keys()) and new_key in set(state_dict_new.keys()):
#             continue
#         else:
#             print('t5_proj:', key)
#     if key not in set(state_dict_new.keys()):
#         print('not in:', key)
#     if key in set(state_dict_new.keys()):
#         print('In:', key)


#
# #NEXTQA 32 to 4
# url_or_filename = '/scratch_xijun/code/Video/SeViLA/lavis/output/BLIP2/NextQA/QA/blip2_nextqa_instruct_2e-5/checkpoint_best.pth'
# checkpoint = torch.load(url_or_filename, map_location="cpu")
# state_dict = checkpoint["model"]
#
# url_or_filename_T = '/scratch_xijun/code/Video/SeViLA/lavis/output/BLIP2/NextQA/QA/blip2_nextqa_instruct_32f/checkpoint_best.pth'
# checkpoint_T = torch.load(url_or_filename_T, map_location="cpu")
# state_dict_T = checkpoint_T["model"]
#
# new_state_dict = state_dict.copy()
#
#
# for key in state_dict.keys():
#     if 'Qformer' in key:
#         new_key = key.replace('Qformer', 'Qformer_T')
#         new_state_dict[new_key] = state_dict_T[key]
#         print(key)
#     if 'query_tokens' in key:
#         new_key = key.replace('query_tokens', 'query_tokens_T')
#         new_state_dict[new_key] = state_dict_T[key]
#         print(key)
#     if 't5_proj' in key:
#         new_key = key.replace('t5_proj', 't5_proj_T')
#         new_state_dict[new_key] = state_dict_T[key]
#         print(key)
#
# save_obj = {"model": new_state_dict}
# torch.save(save_obj, 'blvqa_checkpoints/nextqa_blvqa_blip_flanxl_trimmed_32t4T.pth')
#

# #star
# url_or_filename = '/scratch_xijun/code/Video/SeViLA/lavis/output/BLVQA/STAR/QA/blvqa_32f/checkpoint_best.pth'
# checkpoint = torch.load(url_or_filename, map_location="cpu")
# state_dict = checkpoint["model"]
#
# new_state_dict = state_dict.copy()
#
#
# for key in state_dict.keys():
#     if 'Qformer' in key:
#         new_key = key.replace('Qformer', 'Qformer_T')
#         new_state_dict[new_key] = state_dict[key]
#         print(key)
#     if 'query_tokens' in key:
#         new_key = key.replace('query_tokens', 'query_tokens_T')
#         new_state_dict[new_key] = state_dict[key]
#         print(key)
#     if 't5_proj' in key:
#         new_key = key.replace('t5_proj', 't5_proj_T')
#         new_state_dict[new_key] = state_dict[key]
#         print(key)
#
# save_obj = {"model": new_state_dict}
# torch.save(save_obj, 'blvqa_checkpoints/blvqa_blip_flanxl_trimmed_T.pth')

# #STAR 4 frames
# url_or_filename = '/scratch_xijun/code/Video/SeViLA/lavis/output/BLVQA/STAR/QA/blvqa_4f_2e-5_1000wu/checkpoint_best.pth'
# checkpoint = torch.load(url_or_filename, map_location="cpu")
# state_dict = checkpoint["model"]
#
# new_state_dict = state_dict.copy()
#
#
# for key in state_dict.keys():
#     if 'Qformer' in key:
#         new_key = key.replace('Qformer', 'Qformer_T')
#         new_state_dict[new_key] = state_dict[key]
#         print(key)
#     if 'query_tokens' in key:
#         new_key = key.replace('query_tokens', 'query_tokens_T')
#         new_state_dict[new_key] = state_dict[key]
#         print(key)
#     if 't5_proj' in key:
#         new_key = key.replace('t5_proj', 't5_proj_T')
#         new_state_dict[new_key] = state_dict[key]
#         print(key)
#
# save_obj = {"model": new_state_dict}
# torch.save(save_obj, 'blvqa_checkpoints/star_blvqa_blip_flanxl_trimmed_T_4f.pth')




# #VLEP 32 to 4
# url_or_filename = '/scratch_xijun/code/Video/SeViLA/lavis/output/BLVQA/VLEP/QA/blvqa_4f_2e-5_1000wu_bs16/checkpoint_best.pth'
# checkpoint = torch.load(url_or_filename, map_location="cpu")
# state_dict = checkpoint["model"]
#
# url_or_filename_T = '/scratch_xijun/code/Video/SeViLA/lavis/output/BLVQA/VLEP/QA/blvqa_32f/checkpoint_best.pth'
# checkpoint_T = torch.load(url_or_filename_T, map_location="cpu")
# state_dict_T = checkpoint_T["model"]
#
# new_state_dict = state_dict.copy()
#
#
# for key in state_dict.keys():
#     if 'Qformer' in key:
#         new_key = key.replace('Qformer', 'Qformer_T')
#         new_state_dict[new_key] = state_dict_T[key]
#         print(key)
#     if 'query_tokens' in key:
#         new_key = key.replace('query_tokens', 'query_tokens_T')
#         new_state_dict[new_key] = state_dict_T[key]
#         print(key)
#     if 't5_proj' in key:
#         new_key = key.replace('t5_proj', 't5_proj_T')
#         new_state_dict[new_key] = state_dict_T[key]
#         print(key)
#
# save_obj = {"model": new_state_dict}
# torch.save(save_obj, 'blvqa_checkpoints/vlep_blvqa_blip_flanxl_trimmed_32t4T.pth')


#
# #HOW2QA 32 to 4
# url_or_filename = '/scratch_xijun/code/Video/SeViLA/lavis/output/BLVQA/HOW2QA/QA/blvqa_4f_2e-5_500wu/checkpoint_best.pth'
# checkpoint = torch.load(url_or_filename, map_location="cpu")
# state_dict = checkpoint["model"]
#
# url_or_filename_T = '/scratch_xijun/code/Video/SeViLA/lavis/output/BLVQA/HOW2QA/QA/blvqa_8f_2e-5_1400wu_2gi/checkpoint_best.pth'
# checkpoint_T = torch.load(url_or_filename_T, map_location="cpu")
# state_dict_T = checkpoint_T["model"]
#
# new_state_dict = state_dict.copy()
#
#
# for key in state_dict.keys():
#     if 'Qformer' in key:
#         new_key = key.replace('Qformer', 'Qformer_T')
#         new_state_dict[new_key] = state_dict_T[key]
#         print(key)
#     if 'query_tokens' in key:
#         new_key = key.replace('query_tokens', 'query_tokens_T')
#         new_state_dict[new_key] = state_dict_T[key]
#         print(key)
#     if 't5_proj' in key:
#         new_key = key.replace('t5_proj', 't5_proj_T')
#         new_state_dict[new_key] = state_dict_T[key]
#         print(key)
#
# save_obj = {"model": new_state_dict}
# torch.save(save_obj, 'blvqa_checkpoints/how2qa_vlap_blip_flanxl_trimmed_8t4T.pth')


