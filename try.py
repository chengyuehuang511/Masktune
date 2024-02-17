import timm
import torch.nn as nn

model = timm.create_model('resnet50', pretrained=True)

# print(model)

# for name, module in model.named_modules():
#     print(name, type(module))

model_new = nn.Sequential(*list(model.children())[-2:-1])

for name, module in model_new.named_modules():
    print(name, type(module))