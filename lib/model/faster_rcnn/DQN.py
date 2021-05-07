import math, random

import torch
import torch.nn as nn
import numpy as np
from collections import deque
from torch.autograd import Variable, Function

layer1_num = 1024
layer2_num = 512
#candidate_size = 16
#state_size = 16*2048

class ReplayBuffer(object):
	def __init__(self,ReplayMemory,featd):
		
		#self.state_size = state_size
		self.buffer = deque(maxlen=ReplayMemory)
		self.state_size = featd
		#self.candidate_size = candidate_size

	def push(self,state,action, reward, next_state,done,candidate_list):
		#state add one dim
		#print("#########################  start replay ############################")
		#print("state.shape:",state.shape)
		#print("next_state.shape:",next_state.shape)
		#state_size = state.size(0)*state.size(1)
		#print("state_size:",state_size)
		state= state.view(self.state_size)
		state = state.cpu().detach().numpy()
		state = np.expand_dims(state, 0)

		next_state = next_state.view(self.state_size)
		next_state = next_state.cpu().detach().numpy()
		next_state = np.expand_dims(next_state, 0)

		#print("state.shape:",state.shape)
		#print("next_state.shape:",next_state.shape)
		##state = state.view(self.state_size,-1)
		##next_state = next_state.view(self.state_size,-1)
		self.buffer.append((state,action,reward, next_state,done,candidate_list))
		#print("##########################  end replay  ################################")

	def sample(self,batch_size):

		state, action, reward, next_state, done,candidate_list= zip(*random.sample(self.buffer, batch_size))
		'''
		minibatch = random.sample(self.buffer,batch_size)
		state = [data[0] for data in minibatch]
		action = [data[1] for data in minibatch]
		reward = [data[2] for data in minibatch]
		next_state = [data[3] for data in minibatch]
		done = [data[4] for data in minibatch]
		candidate_list = [data[5] for data in minibatch]'''
		return np.concatenate(state),action,reward,np.concatenate(next_state),done,candidate_list

	def __len__(self):
		return len(self.buffer)


class DQN(nn.Module):
	def __init__(self, ACTION_NUM,featd):
		super(DQN,self).__init__()
		self.ACTION_NUM = ACTION_NUM
		#self.candidate_size = candidate_size
		self.state_size = featd

		self.layers = nn.Sequential(
			nn.Linear(self.state_size,layer1_num),
			nn.ReLU(),
			nn.Linear(layer1_num,layer2_num),
			nn.ReLU(),
			nn.Linear(layer2_num,self.ACTION_NUM),
			)

	def forward(self,x):
		return self.layers(x)

	def act(self,state,epsilon,candidate_list):
		#print("*************************** start action ********************************")
		#print("state.shape in act",state.shape)
		#state_size = state.size(0)*state.size(1)
		#print("state.size:",state_size)
		state = state.view(-1,self.state_size)
		#print("state.shape in act",state.shape)
		#print("candidate_list:",candidate_list)
		if random.random()>epsilon:
			#print("taking action")
			#print("state.astype():",state.type())
			#state = torch.FloatTensor(state)
			q_value = self.forward(state)
			#print("q_value.shape:",q_value.shape)
			#print("q_value:",q_value)

			q_value_temp = q_value[0]
			#print("q_value_temp:",q_value_temp)
			q_value_temp = q_value_temp.cpu().detach().numpy()
			q_value_candidate = q_value_temp[candidate_list]
			#print("q_value[0]",q_value[0])
			#print("q_value_temp",q_value_temp)
			#print("q_value_candidate",q_value_candidate)
			#print("max_value",q_value_candidate.max(0)[0].data[0])
			#action = np.argwhere(q_value_temp == q_value_candidate.max(0)[0].data[0])
			#print("the max of q_value_candidate is:",q_value_candidate.max(0))
			q_value_candidate_max = q_value_candidate.max(0)
			
			#print("q_value_temp",q_value_temp)
			#print("q_value_candidate_max:",q_value_candidate_max)
			#action = q_value_temp.index(q_value_candidate_max)
			action = np.argwhere(q_value_temp == q_value_candidate_max)
			#print("action:",action)
			try:
				action = action[0,0]
			except:
				action = random.sample(candidate_list,1)[0]			
			#action = action[0,0]
			#print("action:",action)
		else:
			action = random.sample(candidate_list,1)[0]
			#print("random action:",action)
		#print("************************   end action  ********************************")
		return action


def update_target(current_model,target_model):
	target_model.load_state_dict(current_model.state_dict())
	##print("sucessfully update target model of DQN")

def compute_td_loss(current_model,target_model,replay_buffer,batch_size):
	#print("------------------------------start compute q loss ------------------------------")
	state, action, reward, next_state,done,candidate_list =replay_buffer.sample(batch_size)
	'''
	minibatch = random.sample(replay_buffer,batch_size)
	state = [data[0] for data in minibatch]
	action = [data[1] for data in minibatch]
	reward = [data[2] for data in minibatch]
	nextState = [data[3] for data in minibatch]
	candidate_list = [data[4] for data in minibatch]'''
	action_index = []

	#print("state.type:",state.type)
	#print("next_state.type:",next_state.type)
	#print("action.type:",action.type)
	#print("reward.type:",reward.type)
	#print("done.type:",done.type)
	state      = Variable(torch.FloatTensor(np.float32(state)))
	next_state = Variable(torch.FloatTensor(np.float32(next_state)))
	action     = Variable(torch.LongTensor(action))
	reward     = Variable(torch.FloatTensor(reward))
	done       = Variable(torch.FloatTensor(done))
	#print("candidate_list",candidate_list)
	##candidate_list = Variable(torch.LongTensor(candidate_list))
	#print("candidate_list",candidate_list)

	#if gpu>=0:
	state,next_state = state.cuda(),next_state.cuda()
	action = action.cuda()
	reward = reward.cuda()
	done = done.cuda()

	q_value = current_model(state)
	#print("q_value.shape in compute_td_loss",q_value.shape)	
	#print("q_value",q_value)
	next_q_values = current_model(next_state)
	next_q_state_values = target_model(next_state)
	#print("aciton",action)

	#print("q_value",q_value)
	q_value = q_value.gather(1,action.unsqueeze(1)).squeeze(1)
	#print("q_value",q_value)	
	#print("q_value.shape in compute_td_loss",q_value.shape)
	for j in range(0,batch_size):
		q_value_temp = next_q_values[j]
		q_value_temp = q_value_temp.cpu().detach().numpy()
		q_candidate_value = q_value_temp[candidate_list[j]]
		q_candidate_value_max = q_candidate_value.max(0)
		#print("next_q_values[j]",next_q_values[j])
		#print("q_value_temp",q_value_temp)
		#print("candidate_list[j]",candidate_list[j])
		#print("q_candidate_value_max:",q_candidate_value_max)
		index = np.argwhere(q_value_temp == q_candidate_value_max)
		#print("index is",index)
		try:
			action_index.append(index[0,0])
		except:
			print("state is :",state[j])
			print("next_state is:",next_state[j])
			print("reward is",reward[j])
			print("candidate_list[j]",candidate_list[j])
			print("q_candidate_value_max:",q_candidate_value_max)
			print("q_value_temp",q_value_temp)
			print("index is ",index)
		#print(action_index)
	action_index = torch.LongTensor(action_index)
	action_index = action_index.cuda()
	#print("action_index.type:",action_index.type())
	#print("action_index.shape",action_index.shape)
	#print(action_index)
	#print("next_q_state_values.shape",next_q_state_values.shape)
	#print("action_index.shape",action_index.shape)
	#print("next_q_state_values",next_q_state_values)
	next_q_value = next_q_state_values.gather(1, action_index.unsqueeze(1)).squeeze(1)
	#print("next_q_value",next_q_value)
	#print("reward.shape",reward.shape)
	#print("next_q_value.shape",next_q_value.shape)
	#next_q_value = next_q_value.cuda()
	#print("q_value.shape",q_value.type())
	expected_q_value = reward + 0.9 * next_q_value * (1 - done)
	#print("expected_q_value.shape",expected_q_value.shape)
	#print("q_value.shape",q_value.shape)


	loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()
	#print("loss:",loss)

	#optimizer.zero_grad()
	#loss.backward()
	#optimizer.step()
	#print("------------------------------end compute q loss ----------------------------------------")

	return loss








