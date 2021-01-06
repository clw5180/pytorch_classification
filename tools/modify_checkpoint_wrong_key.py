import torch

state_dict_1 = torch.load('/home/user/pytorch_classification/checkpoints/efficientnet-b3_2021_01_06_21_58_50-best_model.pth.tar')['state_dict']
state_dict_2 = {}
for k, v in state_dict_1.items():
    if k == 'cladssifier.weight':
        continue
        #state_dict_2['classifier.weight'] = v
    elif k == 'cladssifier.bias':
        continue
        #state_dict_2['classifier.bias'] = v
    else:
        state_dict_2[k] = v
state_dict_result = {}
state_dict_result['state_dict'] = state_dict_2

print(111)
torch.save(state_dict_result, 'efficientnet-b3_2021_01_06_21_58_50-best_model.pth.tar')