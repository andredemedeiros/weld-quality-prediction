"""Lecture contrôlée de MAP_DATA_WELD. Aucune statistique n'est apprise ici."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CHEM = ['C','Si','Mn','S','P','Ni','Cr','Mo','V','Cu','Co','W',
        'O','Ti','N','Al','B','Nb','Sn','As','Sb']
PROCESS = ['current','voltage','current_type','polarity','heat_input',
           'interpass_temperature','weld_type','pwht_temperature','pwht_time']
TARGETS = ['yield_strength','tensile_strength','elongation','reduction_area','charpy_energy','temperature_100J']
OTHER = ['yield_strength','tensile_strength','elongation','reduction_area',
         'charpy_temperature','charpy_energy','hardness','fatt50',
         'primary_ferrite','second_phase_ferrite','acicular_ferrite','martensite',
         'carbide_ferrite','weld_id']
NAMES = CHEM + PROCESS + OTHER
UNITS = dict(zip(CHEM, ['wt%']*12+['ppm']*9))
UNITS.update(dict(zip(PROCESS,['A','V','catégorie','catégorie','kJ/mm','°C',
                             'catégorie','°C','h'])))
UNITS.update(dict(zip(OTHER,['MPa','MPa','%','%','°C','J','kg/mm²','°C',
                            '%','%','%','%','%','identifiant'])))
UNITS['temperature_100J']='°C'
CATEGORICAL = ['current_type','polarity','weld_type']

def load_data(censor_fraction=0.5):
    path = ROOT/'data/welddb.data'
    lines = path.read_text().splitlines()
    if any(len(line.split()) != 44 for line in lines if line.strip()):
        raise ValueError('La base doit avoir exactement 44 champs par ligne.')
    raw = pd.read_csv(path, sep=r'\s+', header=None, names=NAMES,
                      na_values=['N'], dtype=str)
    clean = pd.DataFrame(index=raw.index)
    audit = []
    for c in NAMES:
        if c in CATEGORICAL or c == 'weld_id':
            clean[c] = raw[c]
            continue
        val = raw[c].copy()
        censored = val.str.startswith('<', na=False)
        if c in CHEM:
            clean[c+'_censored'] = censored.astype(float)
        val = val.str.replace('<','',regex=False)
        if c == 'N':
            # Ex. 67tot33res: azote total 67 ppm, résiduel 33 ppm.
            # La colonne N désigne le total; la chaîne source reste archivée.
            total_residual = val.str.contains('tot',na=False)
            val.loc[total_residual] = val.loc[total_residual].str.extract(
                r'^(\d+(?:\.\d+)?)tot',expand=False)
        if c == 'interpass_temperature':
            interval = val.eq('150-200').fillna(False)
            clean['interpass_interval'] = interval.astype(float)
            val.loc[interval]='175'
        if c == 'hardness':
            # Les valeurs sont conservées pour l'audit, jamais comme entrée.
            val = val.str.extract(r'^(\d+(?:\.\d+)?)',expand=False)
        num = pd.to_numeric(val, errors='coerce')
        bad = raw[c].notna() & num.isna()
        if bad.any():
            raise ValueError(f'Valeur non interprétée dans {c}: {raw.loc[bad,c].unique()}')
        num.loc[censored] *= censor_fraction
        # Ces seuils contredisent l'échelle habituelle des autres valeurs.
        # Sans provenance, on refuse de deviner une conversion ppm/wt%.
        ambiguous = ((raw[c].eq('<5') if c == 'V' else
                     raw[c].eq('<0.01') if c in ['Ti','Al'] else
                     pd.Series(False,index=raw.index))).fillna(False)
        num.loc[ambiguous] = np.nan
        clean[c] = num
        if censored.any() or ambiguous.any():
            audit.append({'variable':c,'censored':int(censored.sum()),
                          'ambiguous_set_missing':int(ambiguous.sum())})
    clean['weld_type'] = clean.weld_type.replace({'ShMA':'MMA','SAA':'SA',
                                               'NGSAW':'SAW-NG','NGGMA':'GMA-NG'})
    # Les clefs reposent sur les chaînes originales, indépendamment des choix
    # de traitement des seuils. Même composition observée => même groupe.
    clean['composition_group'] = raw[CHEM].fillna('N').agg('|'.join,axis=1)
    # Sensibilité très conservatrice par famille d'auteur (heuristique).
    clean['source_group'] = raw.weld_id.str.split('-').str[0]
    clean['temperature_100J'] = clean.charpy_temperature.where(clean.charpy_energy.eq(100))
    return raw, clean, pd.DataFrame(audit)

def feature_names(target):
    # Les propriétés après soudage, microstructures, dureté et ID sont exclus.
    cols = CHEM + PROCESS + [c+'_censored' for c in CHEM] + ['interpass_interval']
    if target == 'charpy_energy':
        cols += ['charpy_temperature']
    return cols

def eligible(data,target):
    # Charpy est conditionné par la température d'essai; elle n'est pas imputée.
    return (data.charpy_temperature.notna() if target == 'charpy_energy'
            else pd.Series(True,index=data.index))

def describe(raw,data,audit):
    out = ROOT/'results';out.mkdir(exist_ok=True)
    audit.to_csv(out/'censoring_audit.csv',index=False)
    pd.DataFrame([{'column':i+1,'variable':c,'unit':UNITS[c],
                   'observed_raw':int(raw[c].notna().sum()),
                   'missing_pct':100*raw[c].isna().mean(),
                   'unique_raw':raw[c].nunique()}
                  for i,c in enumerate(NAMES)]).to_csv(out/'data_dictionary.csv',index=False)
    data[TARGETS+['charpy_temperature']].describe().T.to_csv(out/'target_description.csv')
    corr=data[TARGETS].corr();corr.to_csv(out/'target_correlations.csv')
    summary={'rows':len(data),'columns':len(raw.columns),
             'composition_groups':data.composition_group.nunique(),
             'source_families':data.source_group.nunique(),
             'exact_duplicate_rows':int(raw.duplicated().sum()),
             'sha256':hashlib.sha256((ROOT/'data/welddb.data').read_bytes()).hexdigest(),
             'targets':{t:{'observed':int(data[t].notna().sum()),
                           'eligible_unlabeled':int((eligible(data,t)&data[t].isna()).sum())}
                        for t in TARGETS},
             'weld_types':data.weld_type.value_counts().to_dict()}
    (out/'data_summary.json').write_text(json.dumps(summary,indent=2))
    return summary
