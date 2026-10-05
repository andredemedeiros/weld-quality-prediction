"""Contrôles des invariants qui protègent la validité scientifique du projet."""
import sys
import unittest
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from data import load_data,feature_names,eligible,TARGETS
from models import Regressor,MissingRateFilter
from run import available_unlabeled

class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.raw,cls.data,cls.audit=load_data()
    def test_strict_shape_and_targets(self):
        self.assertEqual(self.raw.shape,(1652,44))
        self.assertEqual([int(self.data[t].notna().sum()) for t in TARGETS],
                         [780,738,700,705,879,356])
        for target in TARGETS:
            cols=feature_names(target)
            self.assertFalse(set(cols)&set(TARGETS+['hardness','weld_id','fatt50']))
            self.assertFalse(any('ferrite' in c for c in cols))
    def test_censoring_and_nitrogen(self):
        raw=self.raw;d=self.data
        v=raw.N.eq('67tot33res')
        self.assertTrue(v.any());self.assertTrue((d.loc[v,'N']==67).all())
        v=raw.V.eq('<0.0005')
        np.testing.assert_allclose(d.loc[v,'V'],0.00025)
        self.assertTrue(d.loc[raw.V.eq('<5'),'V'].isna().all())
        self.assertTrue((d.loc[raw.V.eq('<5'),'V_censored']==1).all())
    def test_no_unlabeled_charpy_with_known_temperature(self):
        self.assertEqual(int((eligible(self.data,'charpy_energy')&self.data.charpy_energy.isna()).sum()),0)
    def test_group_exclusion_for_all_targets(self):
        d=self.data
        for target in TARGETS:
            ids=d.index[eligible(d,target)&d[target].notna()].to_numpy()
            for tr,te in GroupKFold(5).split(ids,groups=d.loc[ids,'composition_group']):
                test=set(d.loc[ids[te],'composition_group'])
                train=set(d.loc[ids[tr],'composition_group'])
                u=available_unlabeled(d,target,test)
                self.assertFalse(test&train)
                self.assertFalse(test&set(d.loc[u,'composition_group']))
                self.assertTrue(d.loc[u,target].isna().all())
    def test_no_graph_no_unlabeled_effect(self):
        d=self.data;cols=feature_names('yield_strength')
        ids=d.index[d.yield_strength.notna()].to_numpy()[:80]
        u=d.index[d.yield_strength.isna()].to_numpy()[:80]
        p={'alpha':.001,'gamma_factor':.1,'beta':0.}
        supervised=Regressor('RFF supervisé',p).fit(d.loc[ids,cols],d.loc[ids,'yield_strength'])
        ablation=Regressor('RFF graphe',p).fit(d.loc[ids,cols],d.loc[ids,'yield_strength'],d.loc[u,cols])
        np.testing.assert_allclose(supervised.predict(d.loc[ids[:10],cols]),
                                   ablation.predict(d.loc[ids[:10],cols]),rtol=1e-7,atol=1e-7)
    def test_fitted_preprocessing_is_immutable(self):
        d=self.data;cols=feature_names('yield_strength')
        ids=d.index[d.yield_strength.notna()].to_numpy()[:80]
        m=Regressor('Ridge',{'alpha':1.}).fit(d.loc[ids,cols],d.loc[ids,'yield_strength'])
        impute=m.pre.named_transformers_['numeric'].named_steps['impute']
        statistics=impute.statistics_.copy()
        X=d.loc[ids[:5],cols].copy();X['C']=9999
        m.predict(X)
        np.testing.assert_array_equal(impute.statistics_,statistics)

if __name__=='__main__':unittest.main(verbosity=2)
