"""Diagnostics complémentaires des points d'essai Charpy et de leur sélection.

Exécuter après src/run.py. Les paramètres RF sont fixes a priori. Ces scores
exploratoires sont distincts du benchmark imbriqué principal.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from data import ROOT,load_data,feature_names
from models import Regressor
from run import FIXED_RF,metrics

def main():
    _,d,_=load_data();rows=[]
    for task in ['energy_without_28_100','temperature_at_100J']:
        if task=='energy_without_28_100':
            ids=d.index[d.charpy_energy.notna()&~d.charpy_energy.isin([28,100])].to_numpy()
            target='charpy_energy';cols=feature_names(target)
        else:
            ids=d.index[d.charpy_energy.eq(100)&d.charpy_temperature.notna()].to_numpy()
            target='charpy_temperature';cols=feature_names('yield_strength')
        for fold,(tr,te) in enumerate(GroupKFold(5).split(ids,groups=d.loc[ids,'composition_group']),1):
            for model,params in [('Moyenne',{}),('Forêt',FIXED_RF)]:
                m=Regressor(model,params).fit(d.loc[ids[tr],cols],d.loc[ids[tr],target])
                pred=m.predict(d.loc[ids[te],cols])
                rows.append({'task':task,'model':model,'fold':fold,'n_total':len(ids),
                             **metrics(d.loc[ids[te],target],pred)})
    pd.DataFrame(rows).to_csv(ROOT/'results/charpy_checks.csv',index=False)
    oof=pd.read_csv(ROOT/'results/oof_predictions.csv')
    oof=oof[oof.target.eq('charpy_energy')]
    records=[]
    for model,O in oof.groupby('model'):
        for subset,mask in [('energy_100',O.observed.eq(100)),('energy_28',O.observed.eq(28)),
                            ('other_energies',~O.observed.isin([28,100]))]:
            S=O[mask]
            r=metrics(S.observed,S.predicted)
            # R² n'est pas défini pour un sous-ensemble à cible constante.
            if S.observed.nunique()<2:r['R2']=np.nan
            records.append({'model':model,'subset':subset,'n':len(S),**r})
    pd.DataFrame(records).to_csv(ROOT/'results/charpy_oof_subsets.csv',index=False)
    print(pd.DataFrame(rows).groupby(['task','model'])[['MAE','RMSE','R2']].mean().to_string())

if __name__=='__main__':main()
