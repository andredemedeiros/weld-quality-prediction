"""Prédiction sur un CSV de caractéristiques, dans les unités du dictionnaire."""
import argparse
import json
import joblib
import pandas as pd
from data import ROOT,TARGETS,feature_names

def main():
    p=argparse.ArgumentParser()
    p.add_argument('input',help='CSV avec les colonnes de feature_names(target)')
    p.add_argument('--target',choices=TARGETS,default='yield_strength')
    p.add_argument('--output',default='predictions.csv')
    a=p.parse_args();X=pd.read_csv(a.input)
    cols=feature_names(a.target)
    missing=set(cols)-set(X.columns)
    if missing:raise ValueError(f'Colonnes absentes: {sorted(missing)}')
    if a.target=='charpy_energy' and X.charpy_temperature.isna().any():
        raise ValueError('La température d\'essai Charpy doit être renseignée.')
    model=joblib.load(ROOT/f'results/models/{a.target}.joblib')
    X[a.target+'_prediction']=model.predict(X[cols]);X.to_csv(a.output,index=False)
    print(f'{len(X)} prédictions écrites dans {a.output}')
    print('Prédictions exploratoires; vérifier le domaine de validité et réaliser des essais physiques.')

if __name__=='__main__':main()
