from POP909_Dataset_master.data_process.processor import MidiEventProcessor
import pretty_midi as pyd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from os import listdir
from time import time

np.random.seed(10)



########################################################
############           Class LSTM           ############
########################################################
def sigmoid(x: np.ndarray):
    exp_value = np.min([np.exp(x), 1e7 * np.ones(x.shape)], axis=0)
    exp_value = np.max([exp_value, 1e-9 * np.ones(x.shape)], axis=0)
    # print(np.min(exp_value),np.max(exp_value))
    return exp_value/(exp_value+1/exp_value)

def sigmoid_true(x: np.ndarray):
    exp_value = np.min([np.exp(x), 1e7 * np.ones(x.shape)], axis=0)
    return exp_value/(exp_value+1)

def deriv_sigmoid_true(X: np.ndarray):
    return X*(1-X)

def deriv_tanh(X: np.ndarray):
    return (1-np.power(X,2))


class LSTM_music:

    def __init__(self, min_pitch : int, max_pitch: int, m: int):
        input_dim = max_pitch +1 - min_pitch
        output_dim = m
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch + 1
        self.eps = 1e-4
        self.W_all = np.random.normal(0,np.sqrt(2/output_dim),(4,output_dim,output_dim))
        self.U_all = np.random.normal(0,np.sqrt(2/input_dim),(4,output_dim,input_dim))
        self.b_all = np.random.normal(0,np.sqrt(2/input_dim),(4,output_dim,1))
        # self.W_i = np.random.normal(0,np.sqrt(2/output_dim),(output_dim,output_dim))
        # self.U_i = np.random.normal(0,np.sqrt(2/input_dim),(input_dim,output_dim))
        # self.W_f = np.random.normal(0,np.sqrt(2/output_dim),(output_dim,output_dim))
        # self.U_f = np.random.normal(0,np.sqrt(2/input_dim),(input_dim,output_dim))
        # self.W_o = np.random.normal(0,np.sqrt(2/output_dim),(output_dim,output_dim))
        # self.U_o = np.random.normal(0,np.sqrt(2/input_dim),(input_dim,output_dim))
        # self.W_c = np.random.normal(0,np.sqrt(2/output_dim),(output_dim,output_dim))
        # self.U_c = np.random.normal(0,np.sqrt(2/input_dim),(input_dim,output_dim))
        self.h = np.zeros((output_dim,1))
        self.c = np.zeros((output_dim,1))
        self.W_generation = np.random.normal(0,np.sqrt(2/output_dim),(input_dim,output_dim))
        self.b_generation = np.random.normal(0,np.sqrt(2/output_dim),(input_dim,1))
        self.dim_input = input_dim
        self.dim_output = output_dim
        self.m_s = [np.zeros_like(temp) for temp in self.W_all]+[np.zeros_like(temp) for temp in self.b_all]+[np.zeros_like(temp) for temp in self.U_all]+[np.zeros_like(self.W_generation),np.zeros_like(self.b_generation)]
        self.v_s = [np.zeros_like(temp) for temp in self.W_all]+[np.zeros_like(temp) for temp in self.b_all]+[np.zeros_like(temp) for temp in self.U_all]+[np.zeros_like(self.W_generation),np.zeros_like(self.b_generation)]



    def forward_pass_one_time(self, x: np.ndarray):
        f,i,o,c_tilde = self.W_all.dot(self.h)+self.U_all.dot(x)+self.b_all
        i = sigmoid_true(i)
        f = sigmoid_true(f)
        self.o = sigmoid_true(o)
        c_tilde = np.tanh(c_tilde)
        self.c = f*self.c+i*c_tilde
        self.h = self.o*np.tanh(self.c)
        # self.p = sigmoid(self.W_generation.dot(self.h)+self.b_generation)
        return sigmoid(self.W_generation.dot(self.o)+self.b_generation)

    def forward_pass_multiple_times(self, X_times : np.ndarray):
        dim_in, t_max = X_times.shape
        self.h_times = np.zeros((self.dim_output,t_max,1))
        self.c_times = np.zeros((self.dim_output,t_max,1))
        self.c_tilde_times = np.zeros((self.dim_output,t_max,1))
        self.i_times = np.zeros((self.dim_output,t_max,1))
        self.f_times = np.zeros((self.dim_output,t_max,1))
        self.o_times = np.zeros((self.dim_output,t_max,1))
        self.X_times = np.zeros((self.dim_input,t_max,1))
        self.p = np.zeros((dim_in,t_max,1))
        for t in range(t_max):
            x = X_times[:,t].reshape(dim_in,1).copy()
            self.X_times[:,t] = x
            f,i,o,c_tilde = self.W_all.dot(self.h)+self.U_all.dot(x)+self.b_all
            i = sigmoid_true(i)
            f = sigmoid_true(f)
            o = sigmoid_true(o)
            c_tilde = np.tanh(c_tilde)
            self.c = f*self.c+i*c_tilde
            self.h = o*np.tanh(self.c)
            self.i_times[:,t] = i
            self.c_tilde_times[:,t] = c_tilde
            self.c_times[:,t] = self.c.copy()
            self.h_times[:,t] = self.h.copy()
            self.f_times[:,t] = f
            self.o_times[:,t] = o
            self.p[:,t] = sigmoid(self.W_generation.dot(o)+self.b_generation)



    def compute_loss(self, Y: np.ndarray):
        Y = Y.reshape(self.p.shape)
        return np.mean(-(Y*np.log(self.p)+(1-Y)*np.log(1-self.p)))


def softmax(x: np.ndarray):
    return np.exp(x)/np.sum(np.exp(x), axis=0)

def sigmoid(x: np.ndarray):
    exp_value = np.min([np.exp(x),1e7* np.ones(x.shape)], axis=0)
    exp_value = np.max([exp_value,1e-9* np.ones(x.shape)], axis=0)
    # print(np.min(exp_value),np.max(exp_value))
    return exp_value/(exp_value+1/exp_value)


def Generate_music(LSTM_music_network : LSTM_music, n :int, starting_situation: np.ndarray, name: str):
    res = np.zeros((LSTM_music_network.dim_input,n,1),int)
    h = LSTM_music_network.h
    c = LSTM_music_network.c
    vec_x = starting_situation.reshape((len(starting_situation),1))
    for t in range(n):
        f,i,o,c_tilde = LSTM_music_network.W_all.dot(h)+LSTM_music_network.U_all.dot(vec_x)+LSTM_music_network.b_all
        i = sigmoid_true(i)
        f = sigmoid_true(f)
        o = sigmoid_true(o)
        c_tilde = np.tanh(c_tilde)
        c = f*c+i*c_tilde
        h = o*np.tanh(c)
        p = sigmoid(LSTM_music_network.W_generation.dot(o)+LSTM_music_network.b_generation)
        vec_x = np.random.binomial(1,p)
        res[t] = vec_x.copy()
    n_file = len(listdir("Generated_music/"))
    np.savetxt("Generated_music/LSTM_"+name+str(n_file)+"_length_"+str(n)+".csv",res.reshape(LSTM_music_network.dim_input,n),"%i")
    return res

########################################################
############          Gradient              ############
########################################################

def Compute_Gradients(LSTM_network: LSTM_music, Y: np.ndarray):
    t_batch = Y.shape[1]
    Y = Y.reshape(LSTM_network.p.shape)
    grad_s = -2*(Y-LSTM_network.p)/t_batch
    grad_o_from_generation_times = np.einsum(
        'oi,itl->otl', np.transpose(LSTM_network.W_generation), grad_s)
    grad_b_generation = np.sum(grad_s, axis=1)
    grad_W_generation = np.einsum('otl,itl->io', LSTM_network.o_times, grad_s)
    grad_o_times = np.zeros((LSTM_network.dim_output, t_batch, 1))
    grad_c_times = np.zeros((LSTM_network.dim_output, t_batch+1, 1))
    grad_c_tilde_times = np.zeros((LSTM_network.dim_output, t_batch, 1))
    grad_h_times = np.zeros((LSTM_network.dim_output, t_batch+1, 1))
    grad_f_times = np.zeros((LSTM_network.dim_output, t_batch, 1))
    grad_i_times = np.zeros((LSTM_network.dim_output, t_batch, 1))
    for t in range(t_batch-1, -1, -1):
        grad_c_times[:, t+1] = grad_c_times[:, t+1] + (LSTM_network.o_times[:, t]*grad_h_times[:,t+1])*deriv_tanh(np.tanh(LSTM_network.c_times[:,t]))
        grad_c_tilde_times[:, t] = grad_c_times[:, t+1] * \
            LSTM_network.i_times[:, t]
        grad_i_times[:, t] = grad_c_times[:, t+1] * LSTM_network.c_tilde_times[:, t]
        grad_i_times[:, t] = deriv_sigmoid_true(LSTM_network.i_times[:, t])*grad_i_times[:, t]
        grad_f_times[:, t] = grad_c_times[:, t+1]*LSTM_network.c_times[:, t]
        grad_f_times[:, t] = deriv_sigmoid_true(LSTM_network.f_times[:, t])*grad_f_times[:, t]
        grad_o_times[:, t] = grad_o_from_generation_times[:, t] + \
            grad_h_times[:, t+1]*np.tanh(LSTM_network.c_times[:, t])
        grad_o_times[:, t] = deriv_sigmoid_true(LSTM_network.o_times[:, t])*grad_o_times[:, t]
        grad_c_times[:, t] = grad_c_times[:, t+1]*LSTM_network.f_times[:, t]
        grad_c_times[:, t] = deriv_tanh(LSTM_network.c_tilde_times[:, t])*grad_c_tilde_times[:, t]
        grad_h_times[:, t] = np.transpose(LSTM_network.W_all[0]).dot(grad_f_times[:, t]) + \
            np.transpose(LSTM_network.W_all[1]).dot(grad_i_times[:, t]) + \
            np.transpose(LSTM_network.W_all[2]).dot(grad_o_times[:,t]) + \
            np.transpose(LSTM_network.W_all[3]).dot(grad_c_times[:, t])
    grad_W_i = np.einsum('ktl,mtl->km',grad_i_times,LSTM_network.h_times)
    grad_W_c = np.einsum('ktl,mtl->km',grad_c_tilde_times,LSTM_network.h_times)
    grad_W_f = np.einsum('ktl,mtl->km',grad_f_times,LSTM_network.h_times)
    grad_W_o = np.einsum('ktl,mtl->km',grad_o_times,LSTM_network.h_times)
    grad_U_i = np.einsum('ktl,mtl->km',grad_i_times,LSTM_network.X_times)
    grad_U_c = np.einsum('ktl,mtl->km',grad_c_tilde_times,LSTM_network.X_times)
    grad_U_f = np.einsum('ktl,mtl->km',grad_f_times,LSTM_network.X_times)
    grad_U_o = np.einsum('ktl,mtl->km',grad_o_times,LSTM_network.X_times)
    grad_b_i = np.sum(grad_i_times,axis=1)
    grad_b_o = np.sum(grad_o_times,axis=1)
    grad_b_f = np.sum(grad_f_times,axis=1)
    grad_b_c = np.sum(grad_c_times,axis=1)
    return grad_W_f,grad_W_i, grad_W_o, grad_W_c, grad_b_f,grad_b_i, grad_b_o, grad_b_c,grad_U_f,grad_U_i, grad_U_o, grad_U_c, grad_W_generation, grad_b_generation 

def comparison_gradient(eps: float, grad1: np.ndarray, grad2: np.ndarray):
	res_minus =np.sum(np.abs(grad1-grad2))
	res_plus = np.sum(np.abs(grad1))+np.sum(np.abs(grad2))
	return res_minus/np.max([eps,res_plus])




########################################################
############        Evaluation LSTM_music          ############
########################################################

def softmax(x: np.ndarray):
    return np.exp(x)/np.sum(np.exp(x), axis=0)


def Loss_precompute(Output_p: np.ndarray, Y: np.ndarray):
    Y = Y.reshape(Output_p.shape)
    return -np.mean(Y*np.log(Output_p)+(1-Y)*np.log(1-Output_p))

def convert_encoded_text(text: np.ndarray, decoder: np.ndarray):
    return "".join(decoder[text])



########################################################
############             Adam               ############
########################################################


def Adam(LSTM_music_network: LSTM_music, path_file_array: np.ndarray, time_batch: float, n_epoch: int, eta: float, beta_1: float = .9, beta_2: float = 0.999, eps: float = 1e-8):
    loss=[]
    musics = []
    ## initial loss
    n_files = len(path_file_array)
    batch_count = 0
    for i in range(n_epoch):
        order = np.arange(n_files)
        np.random.shuffle(order)
        for j in tqdm(order):
            path = path_file_array[j]
            music_file = pyd.PrettyMIDI(path)
            X = convert_to_2D(music_file,LSTM_music_network.min_pitch,LSTM_music_network.max_pitch)
            end_time = music_file.get_end_time()
            # end_tick = music_file.time_to_tick()
            initial_offset = np.random.rand()*(end_time%time_batch)
            n_batch = int(end_time//time_batch)
            t = initial_offset + time_batch*np.arange(n_batch+1)
            tick_limit_batch = np.array([music_file.time_to_tick(i) for i in t])
            # print(tick_limit_batch)
            for k in (range(n_batch)):
                batch_count+=1
                tick_init_batch = tick_limit_batch[k]
                tick_end_batch = tick_limit_batch[k+1]
                X_batch = X[:,tick_init_batch:tick_end_batch]
                Y_batch = X[:,tick_init_batch+1:tick_end_batch+1]
                # t=time()
                LSTM_music_network.forward_pass_multiple_times(X_batch)
                # print(time()-t)
                # t = time()
                grads = Compute_Gradients(LSTM_music_network,Y_batch)
                m_s = []
                v_s = []
                for l in range(len(grads)):
                    LSTM_music_network.m_s[l] = beta_1* LSTM_music_network.m_s[l] + (1-beta_1) * grads[l]
                    LSTM_music_network.v_s[l] = beta_2* LSTM_music_network.v_s[l] + (1-beta_2) * np.power(grads[l],2)
                    m_s.append(LSTM_music_network.m_s[l]/(1-np.power(beta_1,i+1)))
                    v_s.append(LSTM_music_network.v_s[l]/(1-np.power(beta_2,i+1)))
                # grad_W_f,grad_W_i, grad_W_o, grad_W_c, grad_b_f,grad_b_i, grad_b_o, grad_b_c,grad_U_f,grad_U_i, grad_U_o, grad_U_c, grad_W_generation, grad_b_generation = grads
                for i in range(4):
                    LSTM_music_network.W_all[i] -= eta/(np.sqrt(v_s[i])+LSTM_music_network.eps)*m_s[i]
                    LSTM_music_network.b_all[i] -= eta/(np.sqrt(v_s[4+i])+LSTM_music_network.eps)*m_s[4+i]
                    LSTM_music_network.U_all[i] -= eta/(np.sqrt(v_s[8+i])+LSTM_music_network.eps)*m_s[8+i]
                LSTM_music_network.W_generation -= eta/(np.sqrt(v_s[12])+LSTM_music_network.eps)*m_s[12]
                LSTM_music_network.b_generation -= eta/(np.sqrt(v_s[13])+LSTM_music_network.eps)*m_s[13]
                if(batch_count==1):
                    smooth_loss = LSTM_music_network.compute_loss(Y_batch)
                else:
                    temp = LSTM_music_network.compute_loss(Y_batch)
                    if (np.isnan(temp)):
                        print(np.min(LSTM_music_network.p),np.max(LSTM_music_network.p))
                    smooth_loss = 0.999*smooth_loss + 0.001*LSTM_music_network.compute_loss(Y_batch)
                # print(time()-t)
                # t = time()
                loss.append(smooth_loss)
                if ((batch_count)%1==0):
                    print("After "+str(batch_count)+" updates, loss:",smooth_loss)
    return loss,musics



def Adam_one_file(LSTM_music_network: LSTM_music, path_file: np.ndarray, time_batch: float, n_epoch: int, eta: float, beta_1: float = .9, beta_2: float = 0.999, eps: float = 1e-8):
    loss=[]
    musics = []
    ## initial loss
    music_file = pyd.PrettyMIDI(path_file)
    X = convert_to_2D(music_file,LSTM_music_network.min_pitch,LSTM_music_network.max_pitch)
    batch_count = 0
    ending_time = music_file.get_end_time()
    for i in range(n_epoch):
        # end_tick = music_file.time_to_tick()
        initial_offset = np.random.rand()*(ending_time%time_batch)
        n_batch = int(ending_time//time_batch)
        t = initial_offset + time_batch*np.arange(n_batch+1)
        tick_limit_batch = np.array([music_file.time_to_tick(i) for i in t])
        order = np.arange(n_batch)
        np.random.shuffle(order)
        for k in order:
            batch_count+=1
            tick_init_batch = tick_limit_batch[k]
            tick_end_batch = tick_limit_batch[k+1]
            X_batch = X[:,tick_init_batch:tick_end_batch]
            Y_batch = X[:,tick_init_batch+1:tick_end_batch+1]
            # t=time()
            LSTM_music_network.forward_pass_multiple_times(X_batch)
            # print(time()-t)
            # t = time()
            grads = Compute_Gradients(LSTM_music_network,Y_batch)
            m_s = []
            v_s = []
            for l in range(len(grads)):
                LSTM_music_network.m_s[l] = beta_1* LSTM_music_network.m_s[l] + (1-beta_1) * grads[l]
                LSTM_music_network.v_s[l] = beta_2* LSTM_music_network.v_s[l] + (1-beta_2) * np.power(grads[l],2)
                m_s.append(LSTM_music_network.m_s[l]/(1-np.power(beta_1,i+1)))
                v_s.append(LSTM_music_network.v_s[l]/(1-np.power(beta_2,i+1)))
            # grad_W_f,grad_W_i, grad_W_o, grad_W_c, grad_b_f,grad_b_i, grad_b_o, grad_b_c,grad_U_f,grad_U_i, grad_U_o, grad_U_c, grad_W_generation, grad_b_generation = grads
            for i in range(4):
                LSTM_music_network.W_all[i] -= eta/(np.sqrt(v_s[i])+LSTM_music_network.eps)*m_s[i]
                LSTM_music_network.b_all[i] -= eta/(np.sqrt(v_s[4+i])+LSTM_music_network.eps)*m_s[4+i]
                LSTM_music_network.U_all[i] -= eta/(np.sqrt(v_s[8+i])+LSTM_music_network.eps)*m_s[8+i]
            LSTM_music_network.W_generation -= eta/(np.sqrt(v_s[12])+LSTM_music_network.eps)*m_s[12]
            LSTM_music_network.b_generation -= eta/(np.sqrt(v_s[13])+LSTM_music_network.eps)*m_s[13]
            # print(time()-t)
            # t = time()
            if(batch_count==1):
                smooth_loss = LSTM_music_network.compute_loss(Y_batch)
            else:
                temp = LSTM_music_network.compute_loss(Y_batch)
                if (np.isnan(temp)):
                    print(np.min(LSTM_music_network.p),np.max(LSTM_music_network.p))
                smooth_loss = 0.999*smooth_loss + 0.001*LSTM_music_network.compute_loss(Y_batch)
            # print(time()-t)
            # t = time()
            loss.append(smooth_loss)
            if ((batch_count)%100==0):
                print("After "+str(batch_count)+" updates, loss:",smooth_loss)
            if ((batch_count)%1000==0):
                musics.append((Generate_music(LSTM_music_network,200,X_batch[:,0],str(batch_count)+"_it_from_"+path_file.split("/")[-1])))
    return loss,musics



def convert_to_2D(analyze_file : pyd.PrettyMIDI, min_pitch: int, max_pitch: int):
    n_ticks = analyze_file.time_to_tick(analyze_file.get_end_time())+1
    res = np.zeros((128,n_ticks),int)
    for i in range(len(analyze_file.instruments)):
        for note in analyze_file.instruments[i].notes:
            tick_start = analyze_file.time_to_tick(note.start)
            tick_end = analyze_file.time_to_tick(note.end)
            for j in range(tick_start,tick_end+1):
                res[note.pitch][j]=1
    return res[min_pitch:max_pitch]