"""Self-training réel sur une tâche illustrative relative, sans verdict industriel.

La cible est T observée au point 100 J. Classe 1: T <= médiane des SEULS
labels visibles de l'entraînement. La médiane varie par pli; ce n'est pas un
seuil de conformité. Le test et ses compositions restent totalement exclus.
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,precision_score,recall_score
from data import ROOT,load_data,feature_names
from models import preprocessor

def main():
    _,d,_=load_data();target='temperature_100J';cols=feature_names(target)
    ids=d.index[d[target].notna()].to_numpy();rows=[]
    folds=list(GroupKFold(5).split(ids,groups=d.loc[ids,'composition_group']))
    for fraction in [.25,.5,1.]:
        for fold,(tr,te) in enumerate(folds,1):
            train=ids[tr];test=ids[te]
            testgroups=set(d.loc[test,'composition_group'])
            rng=np.random.default_rng(42+fold)
            groups=d.loc[train,'composition_group'].unique().copy();rng.shuffle(groups)
            keep=set(groups[:max(2,int(len(groups)*fraction))])
            visible=train[d.loc[train,'composition_group'].isin(keep).to_numpy()]
            hidden=train[~d.loc[train,'composition_group'].isin(keep).to_numpy()]
            natural=d.index[d[target].isna()&~d.composition_group.isin(testgroups)].to_numpy()
            u=np.concatenate([hidden,natural])
            assert not set(d.loc[u,'composition_group'])&testgroups
            assert not set(d.loc[visible,'composition_group'])&testgroups
            assert len(u)>0
            threshold=float(d.loc[visible,target].median())
            yl=(d.loc[visible,target]<=threshold).astype(int).to_numpy()
            yt=(d.loc[test,target]<=threshold).astype(int).to_numpy()
            prep=preprocessor(d.loc[visible,cols])
            XL=prep.fit_transform(d.loc[visible,cols]);XU=prep.transform(d.loc[u,cols])
            XT=prep.transform(d.loc[test,cols])
            ys=np.concatenate([yl,np.full(len(u),-1,dtype=int)])
            assert np.count_nonzero(ys==-1)==len(u)
            for name in ['RF supervised','RF self-training']:
                base=RandomForestClassifier(n_estimators=160,min_samples_leaf=3,
                                             max_features=.7,random_state=42,n_jobs=2)
                added=0;iterations=0;termination='supervised'
                if name=='RF self-training':
                    model=SelfTrainingClassifier(estimator=base,threshold=.9,max_iter=10)
                    model.fit(np.vstack([XL,XU]),ys)
                    added=int(np.count_nonzero(model.labeled_iter_>0))
                    iterations=int(model.n_iter_);termination=model.termination_condition_
                else:model=base.fit(XL,yl)
                yp=model.predict(XT)
                rows.append({'fraction_labelled_groups':fraction,'fold':fold,'model':name,
                             'threshold_C':threshold,'n_labelled':len(visible),'n_test':len(test),
                             'n_natural_unlabelled':len(natural),'n_masked_labels':len(hidden),
                             'n_unlabelled_supplied':len(u) if name=='RF self-training' else 0,
                             'n_pseudo_labels_added':added,'iterations':iterations,'termination':termination,
                             'accuracy':accuracy_score(yt,yp),'balanced_accuracy':balanced_accuracy_score(yt,yp),
                             'F1':f1_score(yt,yp,zero_division=0),'precision':precision_score(yt,yp,zero_division=0),
                             'recall':recall_score(yt,yp,zero_division=0)})
    table=pd.DataFrame(rows);table.to_csv(ROOT/'results/self_training_metrics.csv',index=False)
    print(table.groupby(['fraction_labelled_groups','model'])[['accuracy','balanced_accuracy','F1','n_pseudo_labels_added']].mean().round(3).to_string())
    (ROOT/'results/self_training_design.json').write_text(json.dumps({
         'task':'temperature at an observed 100 J point, relative class',
         'positive_class':'T <= training-visible-label median; not industrial quality',
         'threshold_probability':.9,'probability_calibrated':False,
         'tuning':'fixed parameters; no outer test tuning',
         'seed':42,'outer_folds':5,'split':'composition_group',
         'note':'One masking draw per fold/fraction; exploratory, not repeated inference.'},indent=2))

if __name__=='__main__':main()
