import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

#世界意识的组件
class map_actor(nn.Module):
    def __init__(self, input_dim, output_dim):  # 建立网络初始化数据
        super(map_actor, self).__init__()
        self.input_dim = input_dim  # 定义策略网络输入层的神经元节点数量
        self.output_dim = output_dim  # 定义策略网络输出层的神经元节点数量
        self.fc1 = nn.Linear(self.input_dim, 128)  # 定义输入层和第一中间层之间的交互，第一中间层的节点数量为128
        self.fc2 = nn.Linear(128, 128)  # 第一中间层和第二中间层之间的交互，第二中间层的节点数量为128
        self.fc3 = nn.Linear(128, self.output_dim)  # 定义第二中间层和输出层之间的交互


        self.rewards = []  # 用来积累即时奖励
        self.log_probs = []  # 用来积累当前策略下某时刻采取动作的概率

    def forward(self, input,maxoutput):  # 网络正向传播

        x = self.fc1(input)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        for i in range(self.output_dim):
            if x[i] > maxoutput:
                x[i] = maxoutput
            elif x[i] < 0.01:
                x[i] = 0.01
        return x
class map_critic(nn.Module):#建立价值网络
    def __init__(self, input_dim, output_dim):

        super(map_critic, self).__init__()

        self.input_dim = input_dim

        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_dim, 128)

        self.fc2 = nn.Linear(128, 128)

        self.fc3 = nn.Linear(128, self.output_dim)

    def forward(self, input):

        x = self.fc1(input)

        x = F.relu(x)

        x = self.fc2(x)

        x = F.relu(x)

        x = self.fc3(x)
        return x

class map_predict(nn.Module):#建立预测网络
    def __init__(self, input_dim, output_dim):
        super(map_predict, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.output_dim)
    def forward(self, input):

        x = self.fc1(input)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

class map_fact(nn.Module):#建立实际特征提取网络网络
    def __init__(self, input_dim, output_dim):
        super(map_fact, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.output_dim)
    def forward(self, input):

        x = self.fc1(input)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
class map_action_predict(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(map_action_predict, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.output_dim)
    def forward(self, input):

        x = self.fc1(input)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        return x

class map_reward(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(map_reward, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.output_dim)
    def forward(self, input):

        x = self.fc1(input)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
class map_one(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(map_one, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.output_dim)
    def forward(self, input):

        x = self.fc1(input)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=0)
        return x



#世界意识的灵魂核心

class map_spirit(object):
    def __init__(self, earth, wood, fire, metal, water, gamma, learning_rate, cont, yuanshen):
        self.earth = torch.from_numpy(earth)
        self.wood = torch.from_numpy(wood)
        self.fire = torch.from_numpy(fire)
        self.metal = torch.from_numpy(metal)
        self.water = torch.from_numpy(water)
        earth_wood = torch.cat((self.earth, self.wood), 0)
        earth_wood_fire = torch.cat((earth_wood, self.fire), 0)
        earth_wood_fire_metal = torch.cat((earth_wood_fire, self.metal), 0)
        earth_wood_fire_metal_water = torch.cat((earth_wood_fire_metal, self.water), 0)
        energy_dim = earth_wood_fire_metal_water.shape[0]*earth_wood_fire_metal_water.shape[1]


        self.energy = torch.reshape(earth_wood_fire_metal_water,(1, energy_dim)).squeeze().float()
        self.observation_dim_fact = energy_dim

        self.gamma = gamma
        self.learning_rate = learning_rate
        self.map_fact = map_fact(self.observation_dim_fact, 10)
        self.map_actor = map_actor(self.observation_dim_fact+10, 10)
        self.map_action_predict = map_action_predict(20, 10)
        self.map_predict = map_predict(self.observation_dim_fact + 20, 10)
        self.map_reward = map_reward(self.observation_dim_fact+30, 2)
        self.map_critic = map_critic(self.observation_dim_fact+32, 1)
        self.map_one = map_one(self.observation_dim_fact, 5)


        self.map_fact_optimizer = torch.optim.Adam(self.map_fact.parameters(), lr=self.learning_rate)

        self.map_actor_optimizer = torch.optim.Adam(self.map_actor.parameters(), lr=self.learning_rate)

        self.map_action_predict_optimizer = torch.optim.Adam(self.map_action_predict.parameters(), lr=self.learning_rate)
        self.map_predict_optimizer = torch.optim.Adam(self.map_predict.parameters(), lr=self.learning_rate)
        self.map_reward_optimizer = torch.optim.Adam(self.map_reward.parameters(), lr=self.learning_rate)
        self.map_critic_optimizer= torch.optim.Adam(self.map_critic.parameters(), lr=self.learning_rate)
        self.map_one_optimizer = torch.optim.Adam(self.map_one.parameters(), lr=self.learning_rate)
        self.count = cont

        self.yuanshen = yuanshen



    def train(self, ):
        train_one = self.map_one.forward(self.energy)
        one_action_index = torch.max(train_one, 0)[1].numpy()
        probs = torch.max(train_one, 0)[0]
        train_fact = self.map_fact.forward(self.energy)
        actor_in = torch.cat((train_fact, self.energy), 0)
        actor_out = self.map_actor.forward(actor_in, 100)
        predict_in = torch.cat((actor_in, actor_out), 0)
        predict_out = self.map_predict.forward(predict_in)
        reward_in = torch.cat((predict_in, predict_out), 0)
        reward_out = self.map_reward.forward(reward_in)
        Q_value_in = torch.cat((reward_in, reward_out), 0)
        Q_value_out = self.map_critic.forward(Q_value_in)


        if self.count> 1:

            MSE_loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)
            map_one_loss = MSE_loss_fn(self.yuanshen, self.energy).sum()*probs
            self.map_one_optimizer.zero_grad()
            map_one_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.map_one.parameters(), 0.1)
            self.map_one_optimizer.step()


            if one_action_index == 0:#使得特征提取器能够与动作相关
                train_fact_0 = self.map_fact.forward(self.yuanshen)
                actor_in_0 = torch.cat((train_fact_0, self.yuanshen), 0)
                actor_out_0 = self.map_actor.forward(actor_in_0, 100)

                self.map_action_predict_optimizer.zero_grad()
                self.map_fact_optimizer.zero_grad()
                action_predict_in = torch.cat((train_fact.detach(), train_fact_0), 0)
                action_predict = self.map_action_predict.forward(action_predict_in)
                MSE_loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)
                action_predict_loss = MSE_loss_fn(actor_out_0.detach(), action_predict).sum()
                action_predict_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.map_action_predict.parameters(), 0.1)
                torch.nn.utils.clip_grad_norm_(self.map_fact.parameters(), 0.1)
                self.map_action_predict_optimizer.step()  # 反向传播更新策略网络
                self.map_fact_optimizer.step()
            elif one_action_index == 1:
                train_fact_0 = self.map_fact.forward(self.yuanshen)
                actor_in_0 = torch.cat((train_fact_0, self.yuanshen), 0)
                actor_out_0 = self.map_actor.forward(actor_in_0, 100)
                predict_in_0 = torch.cat((actor_in_0, actor_out_0), 0)
                predict_out_0 = self.map_predict.forward(predict_in_0)
                MSE_loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)
                predict_loss = MSE_loss_fn(predict_out_0, train_fact).sum()
                self.map_predict_optimizer.zero_grad()
                predict_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.map_predict.parameters(), 0.1)
                self.map_predict_optimizer.step()


            elif one_action_index == 2:
                train_fact_0 = self.map_fact.forward(self.yuanshen)
                actor_in_0 = torch.cat((train_fact_0, self.yuanshen), 0)
                actor_out_0 = self.map_actor.forward(actor_in_0, 100)
                predict_in_0 = torch.cat((actor_in_0, actor_out_0), 0)
                predict_out_0 = self.map_predict.forward(predict_in_0)
                reward_in_0 = torch.cat((predict_in_0, predict_out_0), 0)
                reward_out_0 = self.map_reward.forward(reward_in_0)
                a1_ = F.softmax(predict_out_0, dim=0)  # 归一化
                a2_ = F.softmax(train_fact, dim=0)  # 归一化
                max, di = torch.max(a1_, 0)  # 取预测特征表示中，最大的元素的索引
                di = di.numpy()
                max_ = a2_[di]  # 根据索引得到相应位置的实际特征表示的元素值
                r_i = max_ - max
                r_i_LOSS = -r_i*torch.tanh(reward_out_0[1])#对自己当前的心情进行评估

                  # 两个元素值的差作为奖励
                self.map_reward_optimizer.zero_grad()


                r_i_LOSS.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.map_reward.parameters(), 0.1)
                self.map_reward_optimizer.step()



            elif one_action_index == 3:
                train_fact_0 = self.map_fact.forward(self.yuanshen)
                actor_in_0 = torch.cat((train_fact_0, self.yuanshen), 0)
                actor_out_0 = self.map_actor.forward(actor_in_0, 100)
                predict_in_0 = torch.cat((actor_in_0, actor_out_0), 0)
                predict_out_0 = self.map_predict.forward(predict_in_0)
                reward_in_0 = torch.cat((predict_in_0, predict_out_0), 0)
                reward_out_0 = self.map_reward.forward(reward_in_0)
                Q_value_in_0 = torch.cat((reward_in_0, reward_out_0), 0)
                Q_value_out_0 = self.map_critic.forward(Q_value_in_0)
                delta = Q_value_out_0-reward_out_0[0]-self.gamma*Q_value_out
                self.map_critic_optimizer.zero_grad()
                delta.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.map_critic.parameters(), 0.1)
                self.map_critic_optimizer.step()

            elif one_action_index == 4:
                train_fact_0 = self.map_fact.forward(self.yuanshen)
                actor_in_0 = torch.cat((train_fact_0, self.yuanshen), 0)
                actor_out_0 = self.map_actor.forward(actor_in_0, 100)
                predict_in_0 = torch.cat((actor_in_0, actor_out_0), 0)
                predict_out_0 = self.map_predict.forward(predict_in_0)
                reward_in_0 = torch.cat((predict_in_0, predict_out_0), 0)
                reward_out_0 = self.map_reward.forward(reward_in_0)
                print(reward_in_0)
                print(reward_out_0)
                Q_value_in_0 = torch.cat((reward_in_0, reward_out_0), 0)
                Q_value_out_0 = self.map_critic.forward(Q_value_in_0)
                actor_loss = -Q_value_out_0
                self.map_actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.map_actor.parameters(), 0.1)
                self.map_actor_optimizer.step()


        self.yuanshen = self.energy.detach()
        return actor_out.detach().numpy(), self.yuanshen















