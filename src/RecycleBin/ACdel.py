import torch
import torch as T
import pdb
import higher

# A = torch.randn(2,3)
# pdb.set_trace()
# A.expand(N, L, L) # specifies new size
# A.repeat(N,1,1) # specifies number of copies

# A.repeat(2, 1)

m = torch.nn.Linear(1, 3)
# input = torch.randn(2, 2, 2)
input = torch.zeros((2, 1), requires_grad=True)

print(input, '\n')


inner_opt = higher.get_diff_optim(T.optim.Adam([input], lr=0.1), [input])

output = m(input)

# pdb.set_trace()

input1, = inner_opt.step((output*output).sum(), params=[input])

output1 = m(input1)
print(output)

pdb.set_trace()

torch.autograd.grad(output1.sum(), input, create_graph=True)
# torch.autograd.grad(output.sum(), input, allow_unused=True)
