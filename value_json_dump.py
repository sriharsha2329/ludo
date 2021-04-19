# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 16:23:44 2021

@author: sriharsha
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 12:46:31 2021

@author: sriharsha
"""
import csv
import json
import numpy as np
import pandas as pd
import pickle

def dice_roll():
    roll=np.random.randint(1,7)
    total_roll=np.array([roll])
    while roll==6:
        roll=np.random.randint(1,7)
        total_roll=np.append(total_roll,roll)
    return total_roll


def board_zero():
    board=dict()
    board['board0']=np.zeros((4,4,57))
    board['board1']=np.zeros((4,4))
    board['board2']=np.zeros((4,4,8))
    board['board3']=np.zeros((4,4,6))
    return board

def start():
    board=board_zero()
    board['board1']=np.ones((4,4))
    return board

stars=[0,8,13,21,26,34,39,47]
def player_pos(player,board,action,roll):
    if board['board1'][player][action]==1:
        if roll==6:
            board['board0'][player][action][0]=1
            board['board1'][player][action]=0
            board['board2'][player][action][0]=1
        else:
            pass
    else:
        piece_selected=np.argmax(board['board0'][player][action])
        if piece_selected+roll<51:
            board['board0'][player][action][piece_selected+roll]=1
            board['board0'][player][action][piece_selected]=0
            if piece_selected in stars:
                big_board=stars.index(piece_selected)
                board['board2'][player][action][big_board]=0
            if piece_selected+roll in stars:
                big_board=stars.index(piece_selected+roll)
                board['board2'][player][action][big_board]=1
        elif ((piece_selected+roll>=51) and (piece_selected+roll<=56)):
            board['board0'][player][action][piece_selected+roll]=1
            board['board3'][player][action][piece_selected+roll-51]=1
            board['board0'][player][action][piece_selected]=0
            if piece_selected-51>=0:
                board['board3'][player][action][piece_selected-51]=0
    return board
                
def all_pieces_pos(board):
    pieces=dict()
    for player in np.arange(4):
        pieces[player]=np.argwhere(board['board0'][player]>0)[:,1]
    return pieces

def kill_pieces(diff):
    cond=dict()
    for i in np.arange(4):
        if diff<4:
            cond=13*i
        else:
            cond=-13*i
    return cond

def check_pieces(player1,player2,board):
    pieces=all_pieces_pos(board)
    cond=kill_pieces(player2-player1)
    for i in pieces[player1]:
        for j in pieces[player2]:
            if i==j+cond:
                if j<51:
                    if j not in stars:
                        action_j=np.argmax(board['board0'][player2][:,j])
                        board['board0'][player2][:,j]=np.zeros(4)
                        board['board1'][player1][action_j]=1
    return board

def finish_pieces(player,board):
    for player2 in np.arange(4):
        board=check_pieces(player,player2,board)
    return board

def winning(board):
    win=np.array([])
    for player in np.arange(4):
        win_player=np.sum(board['board3'][player][:,5])
        win=np.append(win,win_player)
    return win

def block_pieces(player,board):
    pieces=all_pieces_pos(board)
    if len(pieces[player])>1:
        unique,count=np.unique(pieces[player],return_counts=True)
        dup=unique[count>1]
        for i in dup:
            if i<51:
                if i not in stars:
                    return 0
                else:
                    return 1
            else:
                return 1
    else:
        return 1
    

def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def play_game_init():
    board=start()
    count=0
    stop=False
    wins=np.zeros(4)
    board_storages=pd.DataFrame()
    action_storages=dict()
    for i in np.arange(4):
        action_storages[i]=np.array([])
    while (np.sum(winning(board))!=16) and stop==False:
        count=count+1
        for player in np.arange(4):
            win_cond=winning(board)[player]
            if win_cond==4:
                wins[player]=wins[player]+1
            else:
                roll_dice=dice_roll()
                for roll in roll_dice:
                    with open("board_{}".format(player), "w") as fp:
                        pickle.dump(board,fp)
                        fp.write("\n")
                    board_temp=board
                    action=np.random.randint(4)
                    board_temp=player_pos(player,board,action,roll)
                    while block_pieces(player,board_temp)==0:
                        action=np.random.randint(4)
                        board_temp=player_pos(player,board,action,roll)
                    board=board_temp
                    board=finish_pieces(player,board)
                    action_storages[player]=np.append(action_storages[player],action)
    for i in np.arange(4):
        action_storages[i]=one_hot(action_storages[i].astype(int),4)
    print(board_storages[1][0])
    return action_storages,wins

board_storages,action_storages,wins=play_game_init()
board_storages[0]
action_storages

board_storages[0][13]

def reward(player,board,wins):
    pos=np.argwhere(np.argsort(wins)[::-1]==player)
    if np.sum(board['board3'][player][:,5])==4:
        return 100/(pos+1)
    else:
        return 0

        
def value_space(player,board_storages,gamma,wins):
    board_storage=board_storages[player]
    value=np.zeros(len(board_storage))
    for iterate in np.arange(1000):
        for i,v in enumerate(board_storage):
            if i < len(board_storage)-1:
                value[i]=reward(player,board_storage[i],wins)+(gamma*value[i+1])
            else:
                value[i]=reward(player,board_storage[i],wins)
    return value


example=dict()
for i in np.arange(4):
    example[i]=np.array([])
for i in np.arange(4):
    for j in np.arange(8):
        example[i]=np.append(example[i],dict({j:i))
example[0][2]
     









import torch
from torch.utils.data import TensorDataset
train_ds=TensorDataset(board_storage,action_storage)
train_dl=DataLoader()


    

    
