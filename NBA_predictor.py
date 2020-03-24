import pandas as pd
import math
import csv
import random
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score

base_elo = 1600
team_elos = {}
team_stats = {}
X = []
y = []

folder = 'E:/datas/NBAdata'

def initialize_data(Mstat,Ostat,Tstat):
    new_Mstat = Mstat.drop(['RK', 'Arena'], axis=1)
    new_Ostat = Ostat.drop(['RK', 'G', 'MP'], axis=1)
    new_Tstat = Tstat.drop(['RK', 'G', 'MP'], axis=1)

    team_stats1 = pd.merge(new_Mstat, new_Ostat, how='left', on='Team')
    team_stats2 = pd.merge(team_stats1, new_Tstat, how='left', on='Team')
    return team_stats1.set_index('Team', inplace=False, drop=True)

def get_elo(team):
    try:
        return team_elos[team]
    except:
        team_elos[team] = base_elo
        return team_elos[team]

def calc_elo(win_team, lose_team):
    winner_rank = get_elo(win_team)
    loser_rank = get_elo(lose_team)

    rank_diff = winner_rank - loser_rank
    exp = (rank_diff * -1)/400
    odds = 1/(1 + math.pow(10, exp))
    # 根据rank级别修改K值
    if winner_rank < 2100:
        k = 32
    elif winner_rank >= 2100 and winner_rank < 2400:
        k = 24
    else:
        k = 16

    # 更新rank数值
    new_winner_rank = round(winner_rank + (k * (1 - odds)))
    new_loser_rank = round(loser_rank + (k * (0 - odds)))
    return new_winner_rank,new_loser_rank

def build_dataSet(all_data):
    print("Building data set...")
    X = []
    skip = 0
    for index, row in all_data.iterrows():

        Wteam = row['WTeam']
        Lteam = row['']

