import torch.nn as nn

# 方法1
model = nn.Sequential()
model.add_module('fc1', nn.Linear(3, 4))
model.add_module('fc2', nn.Linear(4, 2))
model.add_module('output', nn.Softmax(2))

print(model.eval())

# 方法2
model2 = nn.Sequential(
    nn.Conv2d(1, 20, (5, 5)),
    nn.ReLU(),
    nn.Conv2d(20, 64, (5, 5)),
    nn.ReLU()
)

print(model2.eval())

# 方法3        
model3 = nn.ModuleList([nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)])
print(model3.eval())
