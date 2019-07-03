import torch

m1=torch.FloatTensor([[4]])
m2 = 4*torch.rand(5,1)
X1 = m1 - m2

result = X1 - torch.FloatTensor([2,2])
print(result)

result1 = X1-X1
print(result1)
#result = 2,2부분에서의 집합 result1은 0
#result2는 4,4부분에서의 집합 result3은 1
result2 = m4_4-m2
print(result2)
result3 = result1+1
print(result3)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#for reproducibility
torch.manual_seed(777)
if device=='cuda':
    torch.cpu.manual_seed_all(777)

#X = torch.FloatTensor([[0,0],[0,1],[1,0],[1,1]])
X = result
Y = result1
#Y = torch.FloatTensor([[0]])
#Y = torch.FloatTensor([[0],[1],[1],[0]])
linear1 = torch.nn.Linear(2,3,bias=True)
linear2 = torch.nn.Linear(3,2,bias=True)
linear3 = torch.nn.Linear(2,2,bias=True)
linear4 = torch.nn.Linear(2,2,bias=True)
linear5 = torch.nn.Linear(2,2,bias=True)
linear6 = torch.nn.Linear(2,1,bias=True)
sigmoid = torch.nn.Sigmoid()

model = torch.nn.Sequential(linear1,sigmoid,linear2,sigmoid,linear3,sigmoid,linear4,sigmoid,linear5,sigmoid,linear6,sigmoid).to(device)

criterion = torch.nn.BCELoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for step in range(10001):
    optimizer.zero_grad()
    hypothesis = model(X)

    # cost/loss function
    cost = criterion(hypothesis, Y)
    cost.backward()
    optimizer.step()

    if step%500 == 0:
        print(step, cost.item())
       # for param in enumerate(model.parameters()):
           # print("가중치 : ",param)

    # Accuracy computation
    # True if hypothesis > 0.5 else False
    
with torch.no_grad():
    predicted = (model(X) > 0.5).float()
    accuracy = (predicted == Y).float().mean()

    print('\nHypothesis: ',hypothesis.detach().cpu().numpy(), '\nCorrect: ', predicted.detach().cpu().numpy(),'\nAccuracy: ', accuracy.item())

X = torch.FloatTensor([[0,0],[0,1],[0.5,0.5],[1,1]])
print(model(X))
