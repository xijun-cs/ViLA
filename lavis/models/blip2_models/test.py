import torch

# A = torch.randn(1,3,2,2)
# B = torch.zeros(1,3)
# B[0,0]=1
# B[0,2]=1
#
#
# mask = B.unsqueeze(-1).unsqueeze(-1).bool()
# mask = mask.expand(-1, -1, 2, 2)
# s_v = A[mask]
#
# print(A)
# print(B)
# print(mask)
# print(s_v.reshape(1,2,2,2))
#


A = torch.randn(1,3,2,2)
B = torch.zeros(1,3)
B[0,0]=1
B[0,2]=1
B = B.long().tolist()
print(A)
print(B)

s_v = A[B]


print(s_v)




# x = torch.as_tensor(
#         [
#             [[1, 2, 3, 4, 5], [6, 7, 8, 9, 0], [11, 12, 13, 14, 15]],
#             [[16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]],
#         ]
#     )
# print(x.shape)
# index = [[0, 0, 1], [1, 2, 0]]
# # tensor([[ 6,  7,  8,  9,  0],
# #         [11, 12, 13, 14, 15],
# #         [16, 17, 18, 19, 20]])
# B = x[index]
# print(B.shape)
# print(x[index])
#
# index_t = torch.as_tensor(index)
# # tensor([[ 6,  7,  8,  9,  0],
# #         [11, 12, 13, 14, 15],
# #         [16, 17, 18, 19, 20]])
# x = x.index_select(0, index_t[0])
# x = x[torch.arange(x.shape[0]).unsqueeze(-1), index_t[1].unsqueeze(-1)].squeeze()
#
# print(x)