"""Reproduction: python src/run.py --mode full. Aucun service externe requis."""
import argparse
import itertools
import json
import time
import platform
import sys
import numpy as np
import pandas as pd
import sklearn
import scipy
import xgboost
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold,KFold
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from data import ROOT,CHEM,PROCESS,TARGETS,UNITS,load_data,describe,feature_names,eligible
from models import GRIDS,Regressor

plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                     'savefig.bbox':'tight','figure.dpi':120})
SEED=42

def metrics(y,p):
    return {'MAE':mean_absolute_error(y,p),'RMSE':np.sqrt(mean_squared_error(y,p)),
            'R2':r2_score(y,p)}

def available_unlabeled(data,target,excluded_groups,group='composition_group'):
    return data.index[eligible(data,target)&data[target].isna()
                      &~data[group].isin(excluded_groups)].to_numpy()

def eda(raw,data):
    fig,ax=plt.subplots(1,2,figsize=(9.5,3.1))
    miss=raw.isna().mean().sort_values(ascending=False)
    ax[0].barh(miss.index[:16][::-1],100*miss.iloc[:16][::-1],color='#315d83')
    ax[0].set_xlabel('Valeurs manquantes (%)')
    for t in TARGETS:
        ax[1].hist(data[t].dropna()/data[t].dropna().std(),bins=25,
                   histtype='step',label=t.replace('_',' '))
    ax[1].set_xlabel('Cible divisée par son écart type (sans centrage)')
    ax[1].set_ylabel('Effectif');ax[1].legend(fontsize=6)
    fig.tight_layout();fig.savefig(ROOT/'results/figures/missingness.pdf');plt.close(fig)
    # ACP descriptive, sans cibles et sans catégories. Pas de prédiction via l'ACP.
    cols=[c for c in CHEM+PROCESS if pd.api.types.is_numeric_dtype(data[c])
          and data[c].isna().mean()<=.6]
    A=StandardScaler().fit_transform(SimpleImputer(strategy='median').fit_transform(data[cols]))
    pca=PCA().fit(A);scores=pca.transform(A)
    pd.DataFrame(pca.components_.T,index=cols).to_csv(ROOT/'results/pca_loadings.csv')
    explained=pca.explained_variance_ratio_
    (ROOT/'results/pca_summary.json').write_text(json.dumps({'features':cols,
         'explained_variance_ratio':explained.tolist(),
         'components_80pct':int(np.searchsorted(explained.cumsum(),.8)+1)},indent=2))
    fig,ax=plt.subplots(1,2,figsize=(9.5,2.8))
    for typ in data.weld_type.unique():
        idx=data.weld_type.eq(typ)
        ax[0].scatter(scores[idx,0],scores[idx,1],s=8,alpha=.4,label=typ)
    ax[0].set_xlabel(f'PC1 ({100*explained[0]:.1f} %)')
    ax[0].set_ylabel(f'PC2 ({100*explained[1]:.1f} %)')
    ax[0].legend(fontsize=6,ncol=2)
    ax[1].plot(np.arange(1,len(explained)+1),100*explained.cumsum(),'o-',ms=3,color='#315d83')
    ax[1].axhline(80,color='grey',ls='--');ax[1].set_xlabel('Nombre de composantes')
    ax[1].set_ylabel('Variance cumulée (%)');fig.tight_layout()
    fig.savefig(ROOT/'results/figures/pca.pdf');plt.close(fig)

def tune(data,target,train_ids,outer_test_groups,name,grid,fold,group='composition_group'):
    """Les U des groupes de validation interne sont également exclus."""
    cols=feature_names(target);X=data.loc[train_ids,cols];y=data.loc[train_ids,target]
    groups=data.loc[train_ids,group]
    splits=list(GroupKFold(3).split(X,y,groups))
    best=None;bestscore=np.inf;trials=[]
    for params in grid:
        errors=[]
        for tr,va in splits:
            trids=train_ids[tr];valids=train_ids[va]
            excluded=set(outer_test_groups)|set(data.loc[valids,group])
            uids=available_unlabeled(data,target,excluded,group)
            assert not (set(data.loc[trids,group])&set(data.loc[valids,group]))
            assert not (set(data.loc[uids,group])&excluded)
            model=Regressor(name,params).fit(data.loc[trids,cols],data.loc[trids,target],
                                            data.loc[uids,cols])
            errors.append(mean_absolute_error(data.loc[valids,target],model.predict(data.loc[valids,cols])))
        score=float(np.mean(errors))
        trials.append({'target':target,'model':name,'outer_fold':fold,
                       'params':json.dumps(params),'inner_MAE':score})
        if score<bestscore:bestscore=score;best=params
    return best,bestscore,trials

def nested(data,targets,quick=False):
    rows=[];oof=[];trials=[];importances=[];audit=[]
    for target in targets:
        cols=feature_names(target)
        ids=data.index[eligible(data,target)&data[target].notna()].to_numpy()
        X=data.loc[ids,cols];y=data.loc[ids,target];groups=data.loc[ids,'composition_group']
        outer=list(GroupKFold(5).split(X,y,groups))
        for name,grid in GRIDS.items():
            start=time.perf_counter()
            if quick:grid=grid[:1]
            for fold,(tr,te) in enumerate(outer,1):
                trids=ids[tr];teids=ids[te];tg=set(data.loc[teids,'composition_group'])
                best,inner,search=tune(data,target,trids,tg,name,grid,fold)
                trials+=search
                uids=available_unlabeled(data,target,tg)
                assert not (tg&set(data.loc[trids,'composition_group']))
                assert not (tg&set(data.loc[uids,'composition_group']))
                m=Regressor(name,best).fit(data.loc[trids,cols],data.loc[trids,target],data.loc[uids,cols])
                pred=m.predict(data.loc[teids,cols]);s=metrics(data.loc[teids,target],pred)
                rows.append({'target':target,'model':name,'fold':fold,**s,
                             'n_train':len(trids),'n_test':len(teids),
                             'n_unlabeled_used':m.n_unlabeled_,'inner_MAE':inner,
                             'params':json.dumps(best)})
                audit.append({'target':target,'model':name,'fold':fold,
                              'train_rows':trids.tolist(),'test_rows':teids.tolist(),
                              'unlabeled_rows':uids.tolist() if name=='RFF graphe' else [],
                              'numeric_columns_kept':m.numeric_columns_})
                oof.extend({'row_id':int(i),'target':target,'model':name,'fold':fold,
                            'observed':float(obs),'predicted':float(p),'composition_group':data.loc[i,'composition_group']}
                           for i,obs,p in zip(teids,data.loc[teids,target],pred))
                if name=='Forêt' and target in ['yield_strength','charpy_energy']:
                    rng=np.random.default_rng(SEED+fold)
                    XX=data.loc[teids,cols].copy()
                    for c in cols:
                        # Tous les indicateurs et valeurs d'une même matière
                        # sont permutés conjointement si c est une concentration.
                        if c.endswith('_censored'):continue
                        associated=[c]+([c+'_censored'] if c in CHEM else [])
                        for repeat in range(3):
                            XP=XX.copy();order=rng.permutation(len(XX))
                            XP[associated]=XX[associated].iloc[order].to_numpy()
                            delta=mean_absolute_error(data.loc[teids,target],m.predict(XP))-s['MAE']
                            importances.append({'target':target,'fold':fold,'feature':c,'repeat':repeat,'delta_MAE':delta})
            print(f'{target:18s} | {name:14s} | {time.perf_counter()-start:.1f} s',flush=True)
            pd.DataFrame(rows).to_csv(ROOT/'results/fold_metrics.csv',index=False)
            pd.DataFrame(oof).to_csv(ROOT/'results/oof_predictions.csv',index=False)
    pd.DataFrame(trials).to_csv(ROOT/'results/hyperparameter_search.csv',index=False)
    pd.DataFrame(importances).to_csv(ROOT/'results/permutation_importance.csv',index=False)
    (ROOT/'results/split_audit.json').write_text(json.dumps(audit))
    return pd.DataFrame(rows),pd.DataFrame(oof)

def summarize(rows,oof):
    output=[]
    for (target,name),s in rows.groupby(['target','model']):
        O=oof[(oof.target==target)&(oof.model==name)]
        record={'target':target,'model':name}
        for metric in ['MAE','RMSE','R2']:
            record[metric+'_mean']=s[metric].mean()
            record[metric+'_sd']=s[metric].std(ddof=1)
        record.update({k+'_pooled':v for k,v in metrics(O.observed,O.predicted).items()})
        output.append(record)
    tab=pd.DataFrame(output).sort_values(['target','MAE_mean'])
    tab.to_csv(ROOT/'results/summary_metrics.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(9.5,3))
    for ax,target in zip(axes,['yield_strength','charpy_energy']):
        a=tab[tab.target==target].sort_values('MAE_mean',ascending=False)
        ax.barh(a.model,a.MAE_mean,xerr=a.MAE_sd,color='#315d83',alpha=.85)
        ax.set_xlabel(f'MAE ± écart type des plis ({UNITS[target]})')
        ax.set_title('Limite élastique' if target=='yield_strength' else 'Énergie Charpy')
    fig.tight_layout();fig.savefig(ROOT/'results/figures/comparison.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9.5,3))
    for ax,target in zip(axes,['yield_strength','charpy_energy']):
        best=tab[tab.target==target].iloc[0]['model']
        o=oof[(oof.target==target)&(oof.model==best)]
        ax.scatter(o.observed,o.predicted,s=9,alpha=.4,color='#315d83')
        lo=min(o.observed.min(),o.predicted.min());hi=max(o.observed.max(),o.predicted.max())
        ax.plot([lo,hi],[lo,hi],'k--',lw=.8)
        ax.set_xlabel(f'Valeur observée ({UNITS[target]})');ax.set_ylabel('Prédiction hors pli')
        ax.set_title(f'{target} — {best}')
    fig.tight_layout();fig.savefig(ROOT/'results/figures/predictions.pdf');plt.close(fig)
    return tab

FIXED_RF={'max_depth':None,'min_samples_leaf':3,'max_features':.7}
def sensitivity(data):
    """Comparaisons exploratoires avec paramètres RF fixés a priori, sans tuning."""
    rows=[]
    for target in ['yield_strength','charpy_energy']:
        for protocol in ['composition','rows','source','without_charpy_temperature']:
            if protocol=='without_charpy_temperature' and target!='charpy_energy':continue
            ids=data.index[eligible(data,target)&data[target].notna()].to_numpy()
            cols=feature_names(target)
            if protocol=='without_charpy_temperature':cols=cols[:-1]
            if protocol=='rows':splits=KFold(5,shuffle=True,random_state=SEED).split(ids)
            else:
                g='source_group' if protocol=='source' else 'composition_group'
                splits=GroupKFold(5).split(ids,groups=data.loc[ids,g])
            for fold,(tr,te) in enumerate(splits,1):
                model=Regressor('Forêt',FIXED_RF).fit(data.loc[ids[tr],cols],data.loc[ids[tr],target])
                p=model.predict(data.loc[ids[te],cols])
                rows.append({'target':target,'protocol':protocol,'fold':fold,**metrics(data.loc[ids[te],target],p)})
        for fraction in [0.,1.]:
            _,changed,_=load_data(fraction)
            cols=feature_names(target);ids=changed.index[eligible(changed,target)&changed[target].notna()].to_numpy()
            for fold,(tr,te) in enumerate(GroupKFold(5).split(ids,groups=changed.loc[ids,'composition_group']),1):
                m=Regressor('Forêt',FIXED_RF).fit(changed.loc[ids[tr],cols],changed.loc[ids[tr],target])
                rows.append({'target':target,'protocol':f'censor_fraction_{fraction}','fold':fold,
                             **metrics(changed.loc[ids[te],target],m.predict(changed.loc[ids[te],cols]))})
    pd.DataFrame(rows).to_csv(ROOT/'results/sensitivity_metrics.csv',index=False)
    print('Sensibilités terminées',flush=True)

def label_mask_experiment(data):
    """Même split; 25/50% des compositions labellisées conservées, labels U cachés.

    Expérience contrôlée exploratoire: les paramètres sont fixes, les labels
    masqués restent totalement inaccessibles à l'apprentissage. Les U naturels
    et artificiels sont exclus de tous les groupes de test.
    """
    target='yield_strength';cols=feature_names(target)
    ids=data.index[data[target].notna()].to_numpy();rows=[]
    splits=list(GroupKFold(5).split(ids,groups=data.loc[ids,'composition_group']))
    for fraction in [.25,.5,1.]:
        for fold,(tr,te) in enumerate(splits,1):
            train=ids[tr];test=ids[te];tg=set(data.loc[test,'composition_group'])
            rng=np.random.default_rng(SEED+fold)
            unique=data.loc[train,'composition_group'].unique().copy();rng.shuffle(unique)
            keep=set(unique[:max(2,int(len(unique)*fraction))])
            labelled=train[data.loc[train,'composition_group'].isin(keep).to_numpy()]
            hidden=train[~data.loc[train,'composition_group'].isin(keep).to_numpy()]
            natural=available_unlabeled(data,target,tg)
            uids=np.concatenate([hidden,natural])
            for name,beta in [('RFF supervisé',0.),('RFF graphe',.1)]:
                params={'alpha':.001,'gamma_factor':.1,'beta':beta}
                m=Regressor(name,params).fit(data.loc[labelled,cols],data.loc[labelled,target],data.loc[uids,cols])
                rows.append({'fraction':fraction,'fold':fold,'model':name,'n_labelled':len(labelled),
                             'n_unlabeled_used':m.n_unlabeled_,**metrics(data.loc[test,target],m.predict(data.loc[test,cols]))})
    pd.DataFrame(rows).to_csv(ROOT/'results/label_mask_metrics.csv',index=False)
    print('Expérience de masquage terminée',flush=True)

def fit_final(data,summary):
    import joblib
    dest=ROOT/'results/models';dest.mkdir(exist_ok=True)
    metadata={}
    for target in TARGETS:
        best=summary[summary.target==target].iloc[0]['model']
        ids=data.index[eligible(data,target)&data[target].notna()].to_numpy()
        params,score,_=tune(data,target,ids,set(),best,GRIDS[best],0)
        cols=feature_names(target);uids=available_unlabeled(data,target,set())
        m=Regressor(best,params).fit(data.loc[ids,cols],data.loc[ids,target],data.loc[uids,cols])
        joblib.dump(m,dest/f'{target}.joblib')
        candidates=data.loc[eligible(data,target)&data[target].isna(),cols]
        if len(candidates):
            pd.DataFrame({'row_id':candidates.index,'predicted':m.predict(candidates)}).to_csv(
                ROOT/f'results/unmeasured_{target}_predictions.csv',index=False)
        metadata[target]={'model':best,'params':params,'inner_MAE':score,
                          'note':'Modèle réentraîné sur tous les labels; pas de nouveau score de test.'}
    (dest/'metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['full','quick','eda','summary'],default='full')
    args=parser.parse_args();start=time.perf_counter()
    raw,data,audit=load_data();describe(raw,data,audit)
    if args.mode=='summary':
        summarize(pd.read_csv(ROOT/'results/fold_metrics.csv'),pd.read_csv(ROOT/'results/oof_predictions.csv'));return
    eda(raw,data)
    if args.mode=='eda':return
    rows,oof=nested(data,TARGETS,quick=args.mode=='quick');tab=summarize(rows,oof)
    if args.mode=='full':sensitivity(data);label_mask_experiment(data);fit_final(data,tab)
    (ROOT/'results/environment.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),
         'sklearn':sklearn.__version__,'pandas':pd.__version__,'numpy':np.__version__,
         'scipy':scipy.__version__,'matplotlib':matplotlib.__version__,'xgboost':xgboost.__version__,'seed':SEED,
         'mode':args.mode,'elapsed_seconds':time.perf_counter()-start},indent=2))
    print(tab.to_string(index=False),flush=True)

if __name__=='__main__':main()
