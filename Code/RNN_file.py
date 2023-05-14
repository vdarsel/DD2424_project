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
############           Class RNN            ############
########################################################

class RNN_music:

    def __init__(self, min_pitch : int, max_pitch: int, m: int):
        self.K = max_pitch-min_pitch+1
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch+1
        self.m = m
        self.b = np.zeros((self.m,1))
        self.c = np.zeros((self.K,1))
        self.W = np.random.normal(0,0.1,(self.m,self.m))
        self.U = np.random.normal(0,0.1,(self.m,self.K))
        self.V = np.random.normal(0,0.1,(self.K,self.m))
        self.h0 = np.zeros((self.m,1))
        self.m_s = [np.zeros((self.m,1)),np.zeros((self.K,1)),np.zeros((self.m,self.m)),np.zeros((self.K,self.m)),np.zeros((self.m,self.K))]
        self.v_s = [np.zeros((self.m,1)),np.zeros((self.K,1)),np.zeros((self.m,self.m)),np.zeros((self.K,self.m)),np.zeros((self.m,self.K))]
        self.eps = 1e-4

    def init_matrix (self, b: np.ndarray, c: np.ndarray, W: np.ndarray, V: np.ndarray, U: np.ndarray):
        self.b = b.copy()
        self.c = c.copy()
        self.W = W.copy()
        self.V = V.copy()
        self.U = U.copy()
        self.K = c.shape[0]
        self.m = b.shape[0]
    
    def forward_pass(self, X: np.ndarray):
        tau = len(X[0])
        self.h = np.zeros((self.m,tau,1))
        self.p = np.zeros((self.K,tau,1))
        h0 =self.h0
        # print(X.shape)
        for t in range(tau):
            vec_x = X[:,[t]]
            # print("W:",self.W.shape)
            # print("h0:",self.h0.shape)
            # print("U:",self.U.shape)
            # print("b:",self.b.shape)
            # print("V:",self.V.shape)
            # print("X:",vec_x.shape)
            h0 = np.tanh(self.W.dot(h0)+self.U.dot(vec_x)+self.b)
            # print("h0:",h0.shape)
            proba = sigmoid(self.V.dot(h0)+self.c)
            self.h[:,t] = h0.copy()
            self.p[:,t] = proba.copy()

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

def Generate_Random_RNN_music(RNN_music_network: RNN_music, K: int, m :int):
    b = np.zeros((m,1))
    c = np.zeros((K,1))
    W = np.random.normal(0,0.1,(m,m))
    U = np.random.normal(0,0.1,(m,K))
    V = np.random.normal(0,0.1,(K,m))
    RNN_music_network.init_matrix(b,c,W,V,U)
    return RNN_music_network

def Generate_music(RNN_music_network : RNN_music, n :int, starting_situation: np.ndarray):
    res = np.zeros((RNN_music_network.K,n,1),int)
    h0 = RNN_music_network.h0
    vec_x = starting_situation.reshape((len(starting_situation),1))
    # print(starting_situation.shape)
    # print("W:",RNN_music_network.W.shape)
    # print("h0:",RNN_music_network.h0.shape)
    # print("U:",RNN_music_network.U.shape)
    # print("b:",RNN_music_network.b.shape)
    # print("V:",RNN_music_network.V.shape)
    # print("X:",vec_x.shape)
    for t in range(n):
        h0 = np.tanh(RNN_music_network.W.dot(h0)+RNN_music_network.U.dot(vec_x)+RNN_music_network.b)
        proba = sigmoid(RNN_music_network.V.dot(h0)+RNN_music_network.c)
        # print(proba.shape)
        vec_x = np.random.binomial(1,p=proba)
        # print("X 2:",vec_x.shape)
        res[:,t] = vec_x.copy()
    n_file = len(listdir("Generated_music/"))
    np.savetxt("Generated_music/"+str(n_file)+"_length_"+str(n)+".csv")
    return res

########################################################
############          Gradient              ############
########################################################

def ComputeGradiant(RNN_music_network : RNN_music, X: np.ndarray, Y: np.ndarray):
    tau = len(Y[0])
    G = -2*(Y.reshape(RNN_music_network.p.shape)-RNN_music_network.p)
    # print("G:",G.shape)
    # print("Y:",Y.shape)
    # print("p:",RNN_music_network.p.shape)
    gradV = np.sum([G[:,t].dot(np.transpose(RNN_music_network.h[:,t])) for t in range(tau)], axis=0)
    gradc = np.sum([G[:,t] for t in range(tau)], axis=0)
    gradW = np.zeros((RNN_music_network.m,RNN_music_network.m))
    gradb = np.zeros((RNN_music_network.m,1))
    gradU = np.zeros((RNN_music_network.m,RNN_music_network.K))
    residual = np.zeros((RNN_music_network.m,1))
    for t in range(tau-1,0,-1):
        dL_dht = np.transpose(RNN_music_network.V).dot(G[:,t]) + residual
        dL_dat = dL_dht*(1-np.power(RNN_music_network.h[:,t],2))
        gradb += dL_dat
        gradW += dL_dat.dot(np.transpose(RNN_music_network.h[:,t-1]))
        gradU += dL_dat.dot(np.transpose(X[:,[t]]))
        residual = np.transpose(RNN_music_network.W).dot(dL_dat)
    dL_dht = np.transpose(RNN_music_network.V).dot(G[:,0]) + residual
    dL_dat = dL_dht*(1-np.power(RNN_music_network.h[:,0],2))
    gradb += dL_dat
    gradW += dL_dat.dot(np.transpose(RNN_music_network.h0))
    gradU += dL_dat.dot(np.transpose(X[:,[0]]))
    gradb = np.min([np.max([gradb,-5*np.ones((RNN_music_network.m,1))], axis=0),5*np.ones((RNN_music_network.m,1))], axis=0)
    gradc = np.min([np.max([gradc,-5*np.ones((RNN_music_network.K,1))], axis=0),5*np.ones((RNN_music_network.K,1))], axis=0)
    gradW = np.min([np.max([gradW,-5*np.ones((RNN_music_network.m,RNN_music_network.m))], axis=0),5*np.ones((RNN_music_network.m,RNN_music_network.m))], axis=0)
    gradV = np.min([np.max([gradV,-5*np.ones((RNN_music_network.K,RNN_music_network.m))], axis=0),5*np.ones((RNN_music_network.K,RNN_music_network.m))], axis=0)
    gradU = np.min([np.max([gradU,-5*np.ones((RNN_music_network.m,RNN_music_network.K))], axis=0),5*np.ones((RNN_music_network.m,RNN_music_network.K))], axis=0)
    return gradb, gradc, gradW, gradV, gradU


def comparison_gradient(eps: float, grad1: np.ndarray, grad2: np.ndarray):
	res_minus =np.sum(np.abs(grad1-grad2))
	res_plus = np.sum(np.abs(grad1))+np.sum(np.abs(grad2))
	return res_minus/np.max([eps,res_plus])

def numeric_gradV_compute(RNN_music_network : RNN_music, X: np.ndarray, Y: np.ndarray, h : float):
    numeric_gradV = np.zeros((RNN_music_network.K,RNN_music_network.m))
    for k in tqdm(range(RNN_music_network.K)):
         for m in range(RNN_music_network.m):
              RNN_music_network.V[k][m]+=h
              l2 = Loss(X,Y,RNN_music_network)
              RNN_music_network.V[k][m]-=2*h
              l1 = Loss(X,Y,RNN_music_network)
              RNN_music_network.V[k][m]+=h
              numeric_gradV[k][m] = (l2-l1)/(2*h)
    return numeric_gradV

def numeric_gradU_compute(RNN_music_network : RNN_music, X: np.ndarray, Y_short: np.ndarray, h : float):
    numeric_gradU = np.zeros((RNN_music_network.m,RNN_music_network.K))
    for k in tqdm(range(RNN_music_network.K)):
         for m in range(RNN_music_network.m):
              RNN_music_network.U[m][k]+=h
              l2 = Loss(X,Y_short,RNN_music_network)
              RNN_music_network.U[m][k]-=2*h
              l1 = Loss(X,Y_short,RNN_music_network)
              RNN_music_network.U[m][k]+=h
              numeric_gradU[m][k] = (l2-l1)/(2*h)
    return numeric_gradU

def numeric_gradW_compute(RNN_music_network : RNN_music, X: np.ndarray, Y_short: np.ndarray, h : float):
    numeric_gradW = np.zeros((RNN_music_network.m,RNN_music_network.m))
    for m1 in tqdm(range(RNN_music_network.m)):
         for m2 in range(RNN_music_network.m):
              RNN_music_network.W[m2][m1]+=h
              l2 = Loss(X,Y_short,RNN_music_network)
              RNN_music_network.W[m2][m1]-=2*h
              l1 = Loss(X,Y_short,RNN_music_network)
              RNN_music_network.W[m2][m1]+=h
              numeric_gradW[m2][m1] = (l2-l1)/(2*h)
    return numeric_gradW

def numeric_gradb_compute(RNN_music_network : RNN_music, X: np.ndarray, Y_short: np.ndarray, h : float):
    numeric_gradb = np.zeros((RNN_music_network.m,1))
    for m in tqdm(range(RNN_music_network.m)):
        RNN_music_network.b[m]+=h
        l2 = Loss(X,Y_short,RNN_music_network)
        RNN_music_network.b[m]-=2*h
        l1 = Loss(X,Y_short,RNN_music_network)
        RNN_music_network.b[m]+=h
        numeric_gradb[m] = (l2-l1)/(2*h)
    return numeric_gradb

def numeric_gradc_compute(RNN_music_network : RNN_music, X: np.ndarray, Y_short: np.ndarray, h : float):
    numeric_gradc = np.zeros((RNN_music_network.K,1))
    for k in tqdm(range(RNN_music_network.K)):
        RNN_music_network.c[k]+=h
        l2 = Loss(X,Y_short,RNN_music_network)
        RNN_music_network.c[k]-=2*h
        l1 = Loss(X,Y_short,RNN_music_network)
        RNN_music_network.c[k]+=h
        # print(l2,l1)
        numeric_gradc[k] = (l2-l1)/(2*h)
    return numeric_gradc


def test_gradient(RNN_music_network : RNN_music, X_short: np.ndarray, Y_short: np.ndarray, h : float):
    print(RNN_music_network.h.shape)
    X = np.zeros((len(X_short),RNN_music_network.K,1))
    Y = np.zeros((len(X_short),RNN_music_network.K,1))
    for k in range(len(X_short)):
        X[k][X_short[k]]=1
        Y[k][Y_short[k]]=1
    print(RNN_music_network.h0.shape)
    gradb, gradc, gradW, gradV, gradU, gradh0 = ComputeGradiant(RNN_music_network,X,Y) 
    print("Test gradient of W...")
    numeric_gradW = numeric_gradW_compute(RNN_music_network,X,Y_short,h)
    print("Gap:\t",comparison_gradient(1e-9,gradW,numeric_gradW))
    print("Test gradient of V...")
    numeric_gradV = numeric_gradV_compute(RNN_music_network,X,Y_short,h)
    print("Gap:\t",comparison_gradient(1e-9,gradV,numeric_gradV))
    print("Test gradient of U...")
    numeric_gradU = numeric_gradU_compute(RNN_music_network,X,Y_short,h)
    print("Gap:\t",comparison_gradient(1e-9,gradU,numeric_gradU))
    print("Test gradient of b...")
    numeric_gradb = numeric_gradb_compute(RNN_music_network,X,Y_short,h)
    print("Gap:\t",comparison_gradient(1e-9,gradb,numeric_gradb))
    print("Test gradient of c...")
    numeric_gradc = numeric_gradc_compute(RNN_music_network,X,Y_short,h)
    print("Gap:\t",comparison_gradient(1e-9,gradc,numeric_gradc))

########################################################
############        Evaluation RNN_music          ############
########################################################

def softmax(x: np.ndarray):
    return np.exp(x)/np.sum(np.exp(x), axis=0)

def Loss(X: np.ndarray, Y: np.ndarray, RNN_music_network: RNN_music):
    tau = len(X)
    res = 0
    h = RNN_music_network.h0.copy()
    for t in range(tau):
        h = np.tanh(RNN_music_network.W.dot(h)+RNN_music_network.U.dot(X[:,[t]])+RNN_music_network.b)
        p = sigmoid(RNN_music_network.V.dot(h)+RNN_music_network.c)
        res -= np.mean(Y[:,[t]]*np.log(p)+(1-Y[:,[t]])*np.log(1-p))
    return res/tau

def Loss_precompute(Output_p: np.ndarray, Y: np.ndarray):
    Y = Y.reshape(Output_p.shape)
    return -np.mean(Y*np.log(Output_p)+(1-Y)*np.log(1-Output_p))

def convert_encoded_text(text: np.ndarray, decoder: np.ndarray):
    return "".join(decoder[text])


########################################################
############       AdaGrad function         ############
########################################################


def AdaGrad(RNN_music_network: RNN_music, X_short: np.ndarray, n_length: int, n_epoch: int, eta: float):
    texts=[]
    n_batch_per_epoch = (len(X_short)-1)//n_length
    loss=[]
    ## initial loss
    X_batch = np.zeros((n_length,RNN_music_network.K,1))
    Y_batch = np.zeros((n_length,RNN_music_network.K,1))
    for l in range(n_length):
        X_batch[l][X_short[-n_length-1+l]]=1
    first_loss = Loss(X_batch,X_short[-n_length:], RNN_music_network)
    print("Initial loss:", first_loss)
    loss.append(first_loss)
    texts.append(convert_encoded_text(Generate_music(RNN_music_network,200),RNN_music_network.decoder))
    for i in range(n_epoch):
        for j in range(n_batch_per_epoch):
            X_short_batch = X_short[j*n_length:(j+1)*n_length]
            Y_short_batch = X_short[j*n_length+1:(j+1)*n_length+1]
            X_batch = np.zeros((n_length,RNN_music_network.K,1))
            Y_batch = np.zeros((n_length,RNN_music_network.K,1))
            for l in range(n_length):
                X_batch[l][X_short_batch[l]]=1
                Y_batch[l][Y_short_batch[l]]=1
            RNN_music_network.forward_pass(X_short_batch)
            grads = ComputeGradiant(RNN_music_network,X_batch,Y_batch)
            for l in range(len(grads)):
                RNN_music_network.m_s[l]+= np.power(grads[l],2)
            RNN_music_network.b -= eta/np.sqrt(RNN_music_network.m_s[0]+RNN_music_network.eps)*grads[0]
            RNN_music_network.c -= eta/np.sqrt(RNN_music_network.m_s[1]+RNN_music_network.eps)*grads[1]
            RNN_music_network.W -= eta/np.sqrt(RNN_music_network.m_s[2]+RNN_music_network.eps)*grads[2]
            RNN_music_network.V -= eta/np.sqrt(RNN_music_network.m_s[3]+RNN_music_network.eps)*grads[3]
            RNN_music_network.U -= eta/np.sqrt(RNN_music_network.m_s[4]+RNN_music_network.eps)*grads[4]
            RNN_music_network.h0 = RNN_music_network.h[:,-1].copy()
            if(i+j==0):
                smooth_loss = RNN_music_network.compute_loss(Y_short_batch)
            else:
                smooth_loss = 0.999*smooth_loss + 0.001*RNN_music_network.compute_loss(Y_short_batch)
            loss.append(smooth_loss)
            if ((i*n_batch_per_epoch+j)%100==99):
                print("After "+str(i*n_batch_per_epoch+j+1)+" updates, loss:",smooth_loss)
            if ((i*n_batch_per_epoch+j)%10000==9999):
                texts.append(convert_encoded_text(Generate_music(RNN_music_network,200),RNN_music_network.decoder))
    return loss,texts

########################################################
############             Adam               ############
########################################################


def Adam(RNN_music_network: RNN_music, path_file_array: np.ndarray, time_batch: float, n_epoch: int, eta: float, beta_1: float = .9, beta_2: float = 0.999, eps: float = 1e-8):
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
            X = convert_to_2D(music_file,RNN_music_network.min_pitch,RNN_music_network.max_pitch)
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
                RNN_music_network.forward_pass(X_batch)
                # print(time()-t)
                # t = time()
                grads = ComputeGradiant(RNN_music_network,X_batch,Y_batch)
                # print(time()-t)
                # t = time()
                m_s = []
                v_s = []
                for l in range(len(grads)):
                    RNN_music_network.m_s[l] = beta_1* RNN_music_network.m_s[l] + (1-beta_1) * grads[l]
                    RNN_music_network.v_s[l] = beta_2* RNN_music_network.v_s[l] + (1-beta_2) * np.power(grads[l],2)
                    m_s.append(RNN_music_network.m_s[l]/(1-np.power(beta_1,i+1)))
                    v_s.append(RNN_music_network.v_s[l]/(1-np.power(beta_2,i+1)))
                # print(time()-t)
                # t = time()
                RNN_music_network.b -= eta/(np.sqrt(v_s[0])+RNN_music_network.eps)*m_s[0]
                RNN_music_network.c -= eta/(np.sqrt(v_s[1])+RNN_music_network.eps)*m_s[1]
                RNN_music_network.W -= eta/(np.sqrt(v_s[2])+RNN_music_network.eps)*m_s[2]
                RNN_music_network.V -= eta/(np.sqrt(v_s[3])+RNN_music_network.eps)*m_s[3]
                RNN_music_network.U -= eta/(np.sqrt(v_s[4])+RNN_music_network.eps)*m_s[4]
                # print("before end batch, h0:",RNN_music_network.h0.shape)
                RNN_music_network.h0 = RNN_music_network.h[:,-1].copy()
                # print("end batch, h0:",RNN_music_network.h0.shape)
                # print(time()-t)
                # t = time()
                if(batch_count==1):
                    smooth_loss = RNN_music_network.compute_loss(Y_batch)
                else:
                    temp = RNN_music_network.compute_loss(Y_batch)
                    if (np.isnan(temp)):
                        print(np.min(RNN_music_network.p),np.max(RNN_music_network.p))
                    smooth_loss = 0.999*smooth_loss + 0.001*RNN_music_network.compute_loss(Y_batch)
                # print(time()-t)
                # t = time()
                loss.append(smooth_loss)
                if ((batch_count)%100==0):
                    print("After "+str(batch_count)+" updates, loss:",smooth_loss)
                if ((batch_count)%1000==0):
                    musics.append((Generate_music(RNN_music_network,200,X_batch[:,0])))
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