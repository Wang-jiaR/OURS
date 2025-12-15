import torch

# 加载.pt文件
model = torch.load('/root/autodl-tmp/daple/daple/baple-main/models/ctx_vectors/ctx_plip_kather_s1.pt')

# 检查模型的基本信息
print("Model architecture:")
print(model)

# 如果模型有state_dict，你可以打印出它的内容
if hasattr(model, 'state_dict'):
    print("\nState dictionary keys:")
    for key in model.state_dict().keys():
        print(key)