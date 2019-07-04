#4,4일때

m2=24*torch.rand(10000,2)-12
#1,1일때
m4=24*torch.rand(10000,2)-12
#-4,4일때
m6=24*torch.rand(10000,2)-12

TRY1=torch.zeros(100,2)
TRY2=torch.zeros(100,2)
TRY3=torch.zeros(100,2)

count=0
cnt=0
print("start")
print(len(m2))

while True:
    if(count==100):
        break
    if ((m2[cnt,0]-4)**2 + (m2[cnt,1]-4)**2)<=9:
        TRY1[count,0]=m2[cnt,0]
        TRY1[count,1]=m2[cnt,1]
        count=count+1
    cnt+=1
    
a=torch.Tensor.numpy(TRY1[:,0])
b=torch.Tensor.numpy(TRY1[:,1])

count=0
import torch
from matplotlib import pyplot as plt

cnt=0
while True:
    if(count==100):
        break
    if ((m4[cnt,0]-(-4))**2 + (m4[cnt,1]-4)**2)<=9:
        TRY2[count,0]=m4[cnt,0]
        TRY2[count,1]=m4[cnt,1]
        count +=1
    cnt+=1

c=torch.Tensor.numpy(TRY2[:,0])
d=torch.Tensor.numpy(TRY2[:,1])

count=0
cnt=0
while True:
    if(count==100):
        break
    if ((m6[cnt,0]-1)**2 + (m6[cnt,1]-1)**2)<=9:
        TRY3[count,0]=m6[cnt,0]
        TRY3[count,1]=m6[cnt,1]
        count+=1
    cnt+=1


e=torch.Tensor.numpy(TRY3[:,0])
f=torch.Tensor.numpy(TRY3[:,1])

plt.scatter(a,b)
plt.scatter(c,d)
plt.scatter(e,f)
plt.show()

a_list=[];b_list=[];c_list=[]

for i in range(100):
    print(e[i],",",f[i])

e = torch.FloatTensor(e)
f = torch.FloatTensor(f)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#for reproducibility
torch.manual_seed(777)
if device=='cuda':
    torch.cpu.manual_seed_all(777)


TRY1_Y = torch.ones(100,1)
TRY2_Y = torch.zeros(100,1)
TRY3_Y = torch.ones(100,1)
    
X=torch.cat([TRY1,TRY2],dim=0)
X=torch.cat([X,TRY3])
print(X.shape)
Y = torch.cat([TRY1_Y,TRY2_Y])
Y = torch.cat([Y,TRY3_Y])
print(Y.shape)

linear1 = torch.nn.Linear(2,3,bias=True)
linear2 = torch.nn.Linear(3,4,bias=True)
linear3 = torch.nn.Linear(4,2,bias=True)
linear4 = torch.nn.Linear(2,1,bias=True)
sigmoid = torch.nn.Sigmoid()

model = torch.nn.Sequential(linear1,sigmoid,linear2,sigmoid,linear3,sigmoid,linear4,sigmoid).to(device)

criterion = torch.nn.BCELoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for step in range(100001):
    optimizer.zero_grad()
    hypothesis = model(X)

    # cost/loss function
    cost = criterion(hypothesis, Y)
    cost.backward()
    optimizer.step()

    if step%100 == 0:
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
