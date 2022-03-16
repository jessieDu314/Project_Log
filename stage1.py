# %%%%
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline


from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import log_loss

from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
# %%%% read revlevant data
SeasonResult  = pd.read_csv('1-4.MRegularSeasonCompactResults.csv')
TourneyResult = pd.read_csv('1-5.MNCAATourneyCompactResults.csv')

SeasonDetail  = pd.read_csv('MRegularSeasonDetailedResults.csv')
TourneyDetail = pd.read_csv('MNCAATourneyDetailedResults.csv')

seeds = pd.read_csv('1-3.MNCAATourneySeeds.csv')
rank  = pd.read_csv('4-1.MMasseyOrdinals.csv')

submission = pd.read_csv('1-6.MSampleSubmissionStage1.csv')

# %%%% Extract average score-per-game for each team in each competing season
def score_per_game(SeasonResult):
    team_id      = np.sort(SeasonResult['WTeamID'].unique())    
    Result = SeasonResult.set_index('Season')
    Result['win']  = 1
    Result['lose'] = 0
    win   = Result[['WTeamID','WScore','NumOT','win']].copy().rename(columns = {'WTeamID':'TeamID','WScore':'Score'})
    lose  = Result[['LTeamID','LScore','NumOT','lose']].copy().rename(columns = {'LTeamID':'TeamID','LScore':'Score','lose':'win'})
    plays = pd.concat([win,lose])
    
    SPG     = pd.DataFrame()
    for t in range(len(team_id)):        
        # total number of games played by each team
        GamesPlayed = plays.loc[plays['TeamID'] == team_id[t]].groupby('Season').count().iloc[:,0].rename('GamesPlayed')
        game        = plays.loc[plays['TeamID'] == team_id[t]].groupby('Season').mean('Score')        
        game['TeamID'] = game['TeamID'].astype('int')                
        SPG = SPG.append(pd.concat([game,GamesPlayed],axis = 1))    
    SPG = SPG.sort_values(by = ['Season','TeamID'])
    SPG['Score'] = SPG['Score']/(SPG['NumOT']*5+40)*40
    del SPG['NumOT']
    return SPG 
SPG = score_per_game(SeasonResult)


# %%%% Esctract more detailed data (offensive and defensive efficiency)
# WOR WDR WAst WTO WStl WBlk WPF
def play_efficiency(SeasonDetail):
    team_id      = np.sort(SeasonDetail['WTeamID'].unique())
    Result = SeasonDetail.set_index('Season').drop(columns = ['DayNum','WScore','LScore','WLoc'])

    win    = Result[['WTeamID', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM','WFTA', 
                     'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'NumOT']].copy()
    
    lose   = Result[['LTeamID', 'LFGM','LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 
                     'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF', 'NumOT']].copy()
    
    win.columns  = ['TeamID', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM','FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF', 'NumOT']
    lose.columns = ['TeamID', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM','FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF', 'NumOT']
    
    plays = pd.concat([win,lose])
    
    PlayEff = pd.DataFrame()
    for t in range(len(team_id)):      
        game = plays.loc[plays['TeamID'] == team_id[t]].groupby('Season').mean()
    
        # calculate game efficiency
        off  = (game['OR']+game['Ast'])/(game['NumOT']*5+40)*40
        defe = (game['DR']+game['TO']+game['Stl']+game['Blk']-game['PF'])/(game['NumOT']*5+40)*40
        PE   = pd.concat([off,defe],axis = 1)
        PE['TeamID'] = team_id[t]
        PlayEff = PlayEff.append(PE)
    PlayEff = PlayEff.rename(columns = {0:'offensive_eff',1:'defensive_eff'})
    PlayEff = PlayEff.sort_values(by = ['Season','TeamID'])
    return PlayEff
PE = play_efficiency(SeasonDetail)

        
# %%%% get the ranking for each team
def ranking(rank):
    team_id = np.sort(rank['TeamID'].unique())
    Result  = rank.set_index('Season') 
    TeamRank = pd.DataFrame()
    for t in range(len(team_id)):
        ranks   = Result.loc[Result['TeamID']==team_id[t]]
        max_day = ranks['RankingDayNum'].groupby('Season').max()
        min_day = ranks['RankingDayNum'].groupby('Season').min()
        rank_up = {}
        for m in range(len(max_day)):
            first_day = ranks.loc[(ranks.index == min_day.index[m])&(ranks['RankingDayNum']==min_day.iloc[m]),'OrdinalRank'].mean()
            last_day  = ranks.loc[(ranks.index == max_day.index[m])&(ranks['RankingDayNum']==max_day.iloc[m]),'OrdinalRank'].mean()
            rank_up[max_day.index[m]] = (last_day > first_day)*1
        rank_up = pd.DataFrame.from_dict(rank_up, orient='index').rename(columns = {0:'RankUp'}).rename_axis('Season')        
        rankMean   = ranks[['TeamID','OrdinalRank']].groupby('Season').mean()
        rankMedian = ranks[['OrdinalRank']].groupby('Season').median()
        ranking = pd.merge(pd.merge(rankMean,rankMedian,on = 'Season'),rank_up,on = 'Season').rename(columns = {'OrdinalRank_x':'RankMean','OrdinalRank_y':'RankMedian'})
        TeamRank = TeamRank.append(ranking)
        TeamRank = TeamRank.sort_values(by = ['Season','TeamID'])
    return TeamRank
       
TeamRank = ranking(rank)

# %%%% Construct full dataset using submission structure (testing+prediction)
def sub_structure(submission,SeasonResult):
    feature = submission.copy()
    feature['Season']   = feature['ID'].apply(lambda x:int(x[:4]))
    feature['TeamID_1'] = feature['ID'].apply(lambda x:int(x[5:9]))
    feature['TeamID_2'] = feature['ID'].apply(lambda x:int(x[10:]))
    # find actual game results
    win_dta = SeasonResult.loc[SeasonResult['Season'] >= 2016,['Season','WTeamID','LTeamID']]
    win_dta.columns   = ['Season','TeamID_1','TeamID_2']
    win_dta['Result'] = 1
    lose_dta = SeasonResult.loc[SeasonResult['Season'] >= 2016,['Season','WTeamID','LTeamID']]
    lose_dta.columns   = ['Season','TeamID_2','TeamID_1']
    lose_dta['Result'] = 0
    target = pd.concat([win_dta,lose_dta],sort = False).copy()
    target = target[target['TeamID_1']<target['TeamID_2']].reset_index().copy()
    del target['index']    
    target['ID'] = target['Season'].apply(str)+'_'+target['TeamID_1'].apply(str)+'_'+target['TeamID_2'].apply(str)
    target = target.drop(['Season','TeamID_2','TeamID_1'],axis = 1).copy()
    feature = pd.merge(feature,target,how = 'left',on = 'ID') 
    feature = feature.drop(['Pred'],axis = 1)
    return feature

target = sub_structure(submission,SeasonResult)


# %%%% Creating Training DataSet
def actualGame(SeasonResult):
    win_dta = SeasonResult.loc[SeasonResult['Season'] >= 2016,['Season','WTeamID','LTeamID']]
    win_dta.columns   = ['Season','TeamID_1','TeamID_2']
    win_dta['Result'] = 1
    lose_dta = SeasonResult.loc[SeasonResult['Season'] >= 2016,['Season','WTeamID','LTeamID']]
    lose_dta.columns   = ['Season','TeamID_2','TeamID_1']
    lose_dta['Result'] = 0
    target = pd.concat([win_dta,lose_dta],sort = False).copy()
    target = target[target['TeamID_1']<target['TeamID_2']]
    target['ID'] = target['Season'].apply(str)+'_'+target['TeamID_1'].apply(str)+'_'+target['TeamID_2'].apply(str)
    target = target.loc[target['Season']<=2021].copy()
    return target
trainingSet = actualGame(SeasonResult)

                                          
# %%%% Merge predicting metrics to training set
def prep_feature(SPG,PE,TeamRank,trainingSet):
    # prepare feature
    SPG_1 = SPG.iloc[SPG.index >= 2016].copy()

    PE_1  = PE.iloc[PE.index >= 2016].copy()
    PE_1['TeamID'] = PE_1['TeamID'].astype('int').copy()
    
    TeamRank_1 = TeamRank.iloc[TeamRank.index >= 2016].copy()
    TeamRank_1['TeamID'] = TeamRank_1['TeamID'].astype('int').copy()
        
    feature = pd.merge(pd.merge(SPG_1,PE_1,how = 'left',on = ['Season','TeamID']),TeamRank_1,how = 'left',on = ['Season','TeamID'])
    feature.columns = ['TeamID', 'Score_1', 'win_1', 'GamesPlayed', 'offensive_eff_1',
                       'defensive_eff_1', 'RankMean_1', 'RankMedian_1', 'RankUp_1']
    # merge training with feature
    team1 = pd.merge(trainingSet,feature, how = 'left',left_on = ('Season','TeamID_1'),right_on = ('Season','TeamID'))
    team1 = team1.drop(['TeamID','GamesPlayed'],axis = 1)
    feature.columns = ['TeamID', 'Score_2', 'win_2', 'GamesPlayed', 'offensive_eff_2',
                       'defensive_eff_2', 'RankMean_2', 'RankMedian_2', 'RankUp_2']
    team1 = pd.merge(team1,feature, how = 'left',left_on = ('Season','TeamID_2'),right_on = ('Season','TeamID'))
    team1 = team1.drop(['TeamID','GamesPlayed'],axis = 1)
    team1['Avg_Score_diff']  = team1['Score_1']-team1['Score_2']
    team1['Win_prob_diff']   = team1['win_1'] - team1['win_2']
    team1['OffensiveE_diff'] = team1['offensive_eff_1'] - team1['offensive_eff_2']
    team1['defensiveE_diff'] = team1['defensive_eff_1'] - team1['defensive_eff_2']
    team1['RankMean_diff']   = team1['RankMean_1'] - team1['RankMean_2'] 
    team1['RankMedian_diff'] = team1['RankMedian_1'] - team1['RankMedian_2'] 
    team1['RankUp']          = 2*team1['RankUp_1'] - team1['RankUp_2']
    team1 = team1.drop(['Score_1', 'win_1','offensive_eff_1', 'defensive_eff_1', 'RankMean_1', 'RankMedian_1','RankUp_1',
                        'Score_2', 'win_2', 'offensive_eff_2', 'defensive_eff_2','RankMean_2', 'RankMedian_2', 'RankUp_2'],axis = 1)
    return team1

trainingFeature = prep_feature(SPG,PE,TeamRank,trainingSet)

# %%%% Merge predicting metric to testing set
def prep_feature(SPG,PE,TeamRank,target):
    testing = target.loc[target['Result'].isna()==False]
    # prepare feature
    SPG_1 = SPG.loc[SPG.index >= 2016].copy()

    PE_1  = PE.loc[PE.index >= 2016].copy()
    PE_1['TeamID'] = PE_1['TeamID'].astype('int').copy()
    
    TeamRank_1 = TeamRank.loc[TeamRank.index >= 2016].copy()
    TeamRank_1['TeamID'] = TeamRank_1['TeamID'].astype('int').copy()
        
    feature = pd.merge(pd.merge(SPG_1,PE_1,how = 'left',on = ['Season','TeamID']),TeamRank_1,how = 'left',on = ['Season','TeamID'])
    feature.columns = ['TeamID', 'Score_1', 'win_1', 'GamesPlayed', 'offensive_eff_1',
                       'defensive_eff_1', 'RankMean_1', 'RankMedian_1', 'RankUp_1']
    # merge testing with feature
    team1 = pd.merge(testing,feature, how = 'left',left_on = ('Season','TeamID_1'),right_on = ('Season','TeamID'))
    team1 = team1.drop(['TeamID','GamesPlayed'],axis = 1)
    feature.columns = ['TeamID', 'Score_2', 'win_2', 'GamesPlayed', 'offensive_eff_2',
                       'defensive_eff_2', 'RankMean_2', 'RankMedian_2', 'RankUp_2']
    team1 = pd.merge(team1,feature, how = 'left',left_on = ('Season','TeamID_2'),right_on = ('Season','TeamID'))
    team1 = team1.drop(['TeamID','GamesPlayed'],axis = 1)
    team1['Avg_Score_diff']  = team1['Score_1']-team1['Score_2']
    team1['Win_prob_diff']   = team1['win_1'] - team1['win_2']
    team1['OffensiveE_diff'] = team1['offensive_eff_1'] - team1['offensive_eff_2']
    team1['defensiveE_diff'] = team1['defensive_eff_1'] - team1['defensive_eff_2']
    team1['RankMean_diff']   = team1['RankMean_1'] - team1['RankMean_2'] 
    team1['RankMedian_diff'] = team1['RankMedian_1'] - team1['RankMedian_2'] 
    team1['RankUp']          = 2*team1['RankUp_1'] - team1['RankUp_2']
    team1 = team1.drop(['Score_1', 'win_1','offensive_eff_1', 'defensive_eff_1', 'RankMean_1', 'RankMedian_1','RankUp_1',
                        'Score_2', 'win_2', 'offensive_eff_2', 'defensive_eff_2','RankMean_2', 'RankMedian_2', 'RankUp_2'],axis = 1)
    return team1
testingFeature = prep_feature(SPG,PE,TeamRank,target)

# %%%%  Building Models
X_train = trainingFeature[['Avg_Score_diff','Win_prob_diff', 'OffensiveE_diff', 'defensiveE_diff', 
                          'RankMean_diff','RankMedian_diff', 'RankUp']].copy()
y_train = trainingFeature['Result']

X_test = testingFeature[['Avg_Score_diff','Win_prob_diff', 'OffensiveE_diff', 'defensiveE_diff', 
                          'RankMean_diff','RankMedian_diff', 'RankUp']].copy()
y_test = testingFeature['Result']

# logistic regression with l2 penalty
def logitReg(X_train,y_train,X_test,y_test):
    nmc     = 250
    shuffle = ShuffleSplit(n_splits=nmc,test_size=0.25)
    alpha   = np.arange(0,4.1,0.1)
    valid_score = []
    for a in alpha:
        if a == 0:
            lr     = LogisticRegression(solver = "lbfgs",max_iter=1000)
            CVInfo = cross_validate(lr, X_train, y_train, cv = shuffle)            
        else:
            fullModel = Pipeline([("scaler",MinMaxScaler()),
                                  ("lr",LogisticRegression(penalty = "l2",C = a,solver = "lbfgs",max_iter=1000))])
            CVInfo    = cross_validate(fullModel, X_train, y_train, cv = shuffle)
        valid_score.append(np.mean(CVInfo['test_score']))
    optimal   = valid_score.index(max(valid_score))
    optimal_a = alpha[optimal]
    a = optimal_a
    if a == 0:
        model = lr
    else:
        model = fullModel        
    model.fit(X_train,y_train)
    logit_pred  = model.predict_proba(X_test)[:,1]
    logit_score = log_loss(y_test, logit_pred, eps=1e-15, normalize=True, sample_weight=None, labels=None)    
    return a,logit_score,logit_pred
logisticCls = logitReg(X_train,y_train,X_test,y_test)

#  stochastic gradient classifier with early stopping using logit loss function
def SGD(X_train,y_train,X_test,y_test):
    fullModel = Pipeline([('scaler',MinMaxScaler()),
                          ('sgd',SGDClassifier(max_iter=5000,penalty = 'none',loss="log", eta0 = 0.01,learning_rate = "adaptive",
                                               early_stopping=True,validation_fraction = 1./3.))])
    model     = fullModel.fit(X_train,y_train)
    sgd_pred  = model.predict_proba(X_test)[:,1]
    sgd_score = log_loss(y_test, sgd_pred, eps=1e-15, normalize=True, sample_weight=None, labels=None)
    return sgd_score,sgd_pred
sgdCls = SGD(X_train,y_train,X_test,y_test)






            
    
# %%%% Prediction of 2022
sub_stage2 = pd.read_csv('MSampleSubmissionStage2.csv')
target2 = sub_stage2.copy()
target2['Season']   = target2['ID'].apply(lambda x:int(x[:4]))
target2['TeamID_1'] = target2['ID'].apply(lambda x:int(x[5:9]))
target2['TeamID_2'] = target2['ID'].apply(lambda x:int(x[10:]))
target2 = target2.drop(['Pred'],axis = 1)
predFeature = prep_feature(SPG,PE,TeamRank,target2)
X_2         = predFeature[['Avg_Score_diff','Win_prob_diff', 'OffensiveE_diff', 'defensiveE_diff', 
                           'RankMean_diff','RankMedian_diff', 'RankUp']].copy()

X_1 = pd.concat([X_train,X_test])
y_1 = pd.concat([y_train,y_test])

fullModel = Pipeline([('scaler',MinMaxScaler()),
                      ('sgd',SGDClassifier(max_iter=5000,penalty = 'none',loss="log", eta0 = 0.01,learning_rate = "adaptive",
                                           early_stopping=True,validation_fraction = 1./3.))])
model     = fullModel.fit(X_1,y_1)
sgd_pred  = model.predict_proba(X_2)[:,1]
del sub_stage2['Pred']
sub_stage2['Pred'] = sgd_pred
sub_stage2.to_csv('StageTwo_Submission.csv')
sub_stage2 = sub_stage2.set_index('ID')




