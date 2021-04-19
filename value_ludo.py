# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 12:46:31 2021

@author: sriharsha
"""


import numpy as np
import pandas as pd

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

def player_probability():
    player_probabilities=dict()
    for i in np.arange(4):
        player_probabilities[i]=0.25*np.ones(4)
    return player_probabilities


def start():
    board=board_zero()
    board['board1']=np.ones((4,4))
    return board

stars=[0,8,13,21,26,34,39,47]
def player_pos(player,board,action,roll,player_prob):
    second_chances=0
    if board['board1'][player][action]==1:
        if roll==6:
            board['board0'][player][action][0]=1
            board['board1'][player][action]=0
            board['board2'][player][action][0]=1
            print('{} roll 6'.format(player))
        else:
            player_prob[player][action]=0.0
            if np.sum(player_prob[player])>0:
                player_prob[player]=player_prob[player]/np.sum(player_prob[player])

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
        elif ((piece_selected+roll>=51) and (piece_selected+roll<=56) and (piece_selected!=56)):
            board['board0'][player][action][piece_selected+roll]=1
            board['board3'][player][action][piece_selected+roll-51]=1
            board['board0'][player][action][piece_selected]=0
            if (piece_selected-51>=0):
                board['board3'][player][action][piece_selected-51]=0
            if piece_selected+roll==56:
                second_chances=1
                print('{} won {}'.format(player,np.sum(board['board3'][player][:,5])))
        else:
            player_prob[player][action]=0
            if np.sum(player_prob[player])>0:
                player_prob[player]=player_prob[player]/np.sum(player_prob[player])
    return board,player_prob,second_chances

                
def all_pieces_pos(board):
    pieces=dict()
    for player in np.arange(4):
        pieces[player]=np.argwhere(board['board0'][player]>0)
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
    second_chances=0
    for i,vi in pieces[player1]:
        for j,vj in pieces[player2]:
            if vi==vj+cond:
                if vj<51:
                    if vj not in stars:
                        board['board0'][player2][j][vj]=0
                        board['board1'][player2][j]=1
                        second_chances=1
                        print('{} killed {}'.format(player1,player2))
                        break
        else:
            continue
        break
    return board,second_chances

def finish_pieces(player,board):
    second_chances=0
    for player2 in np.arange(4):
        if player!=player2:
            board,second_chances=check_pieces(player,player2,board)
            if second_chances==1:
                break
    return board,second_chances

def winning(board):
    win=np.array([])
    for player in np.arange(4):
        win_player=np.sum(board['board3'][player][:,5])
        win=np.append(win,win_player)
    return win



def block_pieces(player,board):
    pieces=all_pieces_pos(board)[player]
    pieces=np.array(pieces)[:,1]
    if len(pieces)>1:
        unique,count=np.unique(pieces,return_counts=True)
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
    wins=np.zeros(4)
    board_storages=dict()
    action_storages=dict()
    probability_storages=dict()
    player_prob=player_probability()
    for player in np.arange(4):
        board_storages[player]=np.array([])
        probability_storages[player]=np.array([])
        board_obs=dict()
        for key,value in board.items():
            board_obs[key]=value.flatten()
        board_storages[player]=np.append(board_storages[player],board_obs)
        action_storages[player]=np.array([])
    while (np.sum(winning(board))!=16):
        count=count+1
        print(count)
        for player in np.arange(4):
            second_chances=1
            while second_chances==1:
                second_chances=0
                win_cond=winning(board)[player]
                if win_cond==4:
                    wins[player]=wins[player]+1
                else:
                    roll_dice=dice_roll()
                    for roll in roll_dice:
                        player_prob[player]=player_prob[player]+0.25
                        if np.sum(player_prob[player])>0:
                            player_prob[player]=player_prob[player]/np.sum(player_prob[player])
                        action=np.random.choice(4, p=player_prob[player])
                        board_temp,player_prob,second_chances=player_pos(player,board,action,roll,player_prob)
                        #print('player : {} ,action :{}, player_prob : {}, roll : {}'.format(player,action,player_prob,roll))
                        while (block_pieces(player,board_temp)==0 or player_prob[player][action]==0.0):
                            player_prob[player][action]=0.0
                            if np.sum(player_prob[player])>0:
                                player_prob[player]=player_prob[player]/np.sum(player_prob[player])
                                action=np.random.choice(4, p=player_prob[player])
                                board_temp,player_prob,second_chances=player_pos(player,board,action,roll,player_prob)
                            else:
                                break
                        board=board_temp
                        board,second_chances=finish_pieces(player,board)
                        action_storages[player]=np.append(action_storages[player],action)
                        probability_storages[player]=np.append(probability_storages[player],player_prob[player])
                        board_obs=dict()
                        for key,value in board.items():
                            board_obs[key]=value.flatten()
                        board_storages[player]=np.append(board_storages[player],board_obs)
                        if second_chances==1:
                            break
    for player in np.arange(4):
        action_storages[player]=one_hot(action_storages[player].astype(int),4)
        probability_storages[player]=probability_storages[player].reshape(-1,4)
    return board_storages,action_storages,probability_storages,wins

board_storages,action_storages,probability_storages,wins=play_game_init()


def board_unravel(player,board_obs):
    board=dict()
    board['board0']=board_obs['board0'].reshape(4,4,57)
    board['board1']=board_obs['board1'].reshape(4,4)
    board['board2']=board_obs['board2'].reshape(4,4,8)
    board['board3']=board_obs['board3'].reshape(4,4,6)
    return board


def reward(player,board,wins):
    pos=np.argwhere(np.argsort(wins)[::-1]==player)
    if np.sum(board['board3'][player][:,5])==4:
        return 100/(pos+1)
    else:
        return 0

def probability_zero(prob):
    if np.sum(prob)==0:
        prob=0.25*np.ones(4)
    return prob

def value_space(player,board_storages,gamma,wins,action_storages,probability_storages):
    board_storage=board_storages[player]
    action_storage=action_storages[player]
    probability_storage=probability_storages[player]
    value=np.zeros(len(board_storage))
    for iterate in np.arange(1000):
        for i,v in enumerate(board_storage):
            board=board_unravel(player,board_storage[i])
            if i < len(board_storage)-1:
                prob=probability_zero(probability_storage[i])
                action_prob=np.matmul(action_storage[i],prob)
                value[i]=reward(player,board,wins)+(gamma*action_prob*value[i+1])
            else:
                value[i]=reward(player,board,wins)
    return value

value=value_space(0,board_storages,0.9,wins,action_storages,probability_storages)





