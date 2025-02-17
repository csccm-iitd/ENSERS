
import torch as T
import torch.nn as nn
import pdb

# T.manual_seed_all(0)
T.manual_seed(1)
model = nn.Sequential(
          nn.Linear(3, 3),
          nn.SELU(),
          nn.Linear(3, 1),
          nn.SELU()
        )

for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)

a = T.tensor([3.0], requires_grad=True)
b = T.tensor([2.0, 0.50])

inp = T.cat((a, b))

out = model(inp)
grad = T.autograd.grad(out.sum(), a)
print(grad)