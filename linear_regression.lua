require 'nn'

-- 1. Linear Regression Model
model = nn.Sequential()
model:add(nn.Linear(5, 2))

-- 2. MSE Criterion
crit = nn.MSECriterion()

-- 3. Generate Data
X = torch.rand(5)
T = torch.Tensor({1, 2})

param, grad_param = model:getParameters()

for i=1, 300 do
  O = model:forward(X)
  l = crit:forward(O, T)
  
  grad_param:zero()
  dl_do = crit:backward(O, T)
  model:backward(X, dl_do)

  model:updateParameters(0.01) 
  -- or param:add(-0.01, grad_param)
  
  if i % 50 == 0 then
    print ("[output]")
    print (O)
    
    print ("[loss]")
    print (l)   
  end
end
