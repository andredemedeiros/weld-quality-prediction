"""Modèles inductifs et prétraitement appris sur les seuls labels d'entraînement."""
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor, kneighbors_graph
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_approximation import RBFSampler
from xgboost import XGBRegressor
from data import CATEGORICAL

class MissingRateFilter(BaseEstimator,TransformerMixin):
    """Sélection de variables numériques avec <=60% de valeurs manquantes."""
    def __init__(self,max_missing=0.60):
        self.max_missing=max_missing
    def fit(self,X,y=None):
        self.columns_=list(X.columns[X.isna().mean() <= self.max_missing])
        if not self.columns_:raise ValueError('Aucune variable numérique disponible')
        return self
    def transform(self,X):return X[self.columns_]

def preprocessor(X):
    numeric=[c for c in X if c not in CATEGORICAL]
    nums=Pipeline([('filter',MissingRateFilter()),
                   ('impute',SimpleImputer(strategy='median',add_indicator=True)),
                   ('scale',StandardScaler())])
    cats=Pipeline([('impute',SimpleImputer(strategy='constant',fill_value='missing')),
                   ('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))])
    return ColumnTransformer([('numeric',nums,numeric),('category',cats,CATEGORICAL)],
                             sparse_threshold=0)

GRIDS={
 'Moyenne':[{}],
 'Ridge':[{'alpha':a} for a in [0.1,1,10,100]],
 'kNN':[{'n_neighbors':k,'weights':'distance'} for k in [5,15,30]],
 'SVR RBF':[{'C':c,'epsilon':0.1,'gamma':g} for c in [10,100]
            for g in ['scale',0.01]],
 'Arbre':[{'max_depth':d,'min_samples_leaf':l} for d in [4,10,None] for l in [3,10]],
 'Forêt':[{'max_depth':d,'min_samples_leaf':l,'max_features':0.7}
          for d in [12,None] for l in [1,3]],
 'XGBoost':[{'max_depth':depth,'learning_rate':rate,'reg_lambda':1.0}
             for depth in [2,4] for rate in [0.05,0.1]],
 'RFF supervisé':[{'alpha':a,'gamma_factor':g,'beta':0.0}
                  for a in [0.001,0.01] for g in [0.1,1.0]],
 'RFF graphe':[{'alpha':a,'gamma_factor':g,'beta':b}
               for a in [0.001,0.01] for g in [0.1,1.0] for b in [0.1,1.0]],
}

class Regressor:
    """Interface commune, cibles standardisées; les scores restent en unités natives.

    RFF graphe minimise: ||Z_L theta-y||²/l + alpha ||theta||²
       + beta theta^T Z^T L Z theta / sum(W).
    Z contient 96 features de Fourier et un intercept non pénalisé.
    L est le laplacien non normalisé d'un graphe kNN symétrique pondéré.
    Les U n'entrent que dans le terme de graphe, jamais dans la perte supervisée.
    """
    def __init__(self,name,params,seed=42):
        self.name=name;self.params=params.copy();self.seed=seed
    def fit(self,X,y,X_unlabeled=None):
        self.pre=preprocessor(X)
        A=self.pre.fit_transform(X)
        y=np.asarray(y,float)
        self.mean_=float(y.mean());self.std_=float(y.std()) or 1.
        z=(y-self.mean_)/self.std_
        p=self.params.copy();self.n_unlabeled_=0
        if self.name.startswith('RFF'):
            use_u=(self.name=='RFF graphe' and X_unlabeled is not None and len(X_unlabeled)>0)
            U=self.pre.transform(X_unlabeled) if use_u else np.empty((0,A.shape[1]))
            self.n_unlabeled_=len(U)
            B=np.vstack([A,U]);n=len(B);l=len(A)
            self.rff=RBFSampler(gamma=p['gamma_factor']/A.shape[1],n_components=96,
                                random_state=self.seed)
            F=self.rff.fit_transform(B)
            Z=np.column_stack([F,np.ones(n)])
            R=np.eye(Z.shape[1]);R[-1,-1]=0
            H=Z[:l].T@Z[:l]/l+p['alpha']*R
            if p['beta']>0:
                D=kneighbors_graph(B,n_neighbors=min(10,n-1),mode='distance',
                                   include_self=False)
                dist=D.data.copy()
                bandwidth=np.median(dist[dist>0]) if np.any(dist>0) else 1.
                D.data=np.exp(-dist**2/(2*bandwidth**2))
                W=D.maximum(D.T)
                L=sparse.diags(np.asarray(W.sum(axis=1)).ravel())-W
                H+=p['beta']*(Z.T@(L@Z))/max(float(W.sum()),1.)
            self.theta_=np.linalg.solve(H+1e-10*np.eye(H.shape[0]),Z[:l].T@z/l)
        else:
            constructors={'Moyenne':DummyRegressor,'Ridge':Ridge,
                          'kNN':KNeighborsRegressor,'SVR RBF':SVR,
                          'Arbre':DecisionTreeRegressor,'Forêt':RandomForestRegressor,'XGBoost':XGBRegressor}
            if self.name in ['Arbre','Forêt','XGBoost']:p['random_state']=self.seed
            if self.name=='Forêt':p.update(n_estimators=160,n_jobs=2)
            if self.name=='XGBoost':p.update(n_estimators=160,n_jobs=2,objective='reg:squarederror',tree_method='hist')
            self.model=constructors[self.name](**p).fit(A,z)
        self.numeric_columns_=self.pre.named_transformers_['numeric'].named_steps['filter'].columns_
        return self
    def predict(self,X):
        A=self.pre.transform(X)
        if self.name.startswith('RFF'):
            Z=np.column_stack([self.rff.transform(A),np.ones(len(A))]);z=Z@self.theta_
        else:z=self.model.predict(A)
        return self.mean_+self.std_*z
