"""Feature engineering, baseline supervised models and reproducible evaluation."""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "models"))
REPORT_DIR = Path(os.getenv("REPORT_DIR", ROOT / "reports"))
TARGETS = ["improvement_potential_pct", "electricity_saving_kwh", "gas_saving_kwh", "carbon_reduction_kg_co2e", "value_scenario_low_gbp", "value_scenario_high_gbp", "value_reference_central_gbp"]
NUMERIC = ["age", "floor_area", "bedrooms", "value_current_gbp", "annual_electricity_kwh", "annual_gas_kwh", "energy_intensity_kwh_m2", "space_heat_kwh", "hot_water_kwh", "base_electricity_kwh", "other_gas_kwh", "pv_offset_kwh"]
CATEGORICAL = ["property_type", "energy_rating", "heating", "insulation", "glazing", "region"]
BOOLS = ["solar", "smart_controls", "has_loft", "cavity_walls", "wall_insulated", "draught_proofed", "heat_pump_space", "solar_roof_suitable"]
FEATURES = NUMERIC + CATEGORICAL + BOOLS

def load_data(data_dir=DATA_DIR):
    data_dir=Path(data_dir)
    p=pd.read_csv(data_dir/'properties.csv')
    e=pd.read_csv(data_dir/'energy_monthly.csv') if (data_dir/'energy_monthly.csv').exists() else pd.DataFrame()
    labels=pd.read_csv(data_dir/'reference_labels.csv') if (data_dir/'reference_labels.csv').exists() else pd.DataFrame()
    sales=pd.read_csv(data_dir/'sales_history.csv') if (data_dir/'sales_history.csv').exists() else pd.DataFrame()
    if 'split' not in p.columns: p['split']='train'
    # If monthly records exist, aggregate only observed months. Missing months are not filled with zero.
    if not e.empty and {'property_id','electricity_kwh','gas_kwh'}.issubset(e.columns):
        agg=e.groupby('property_id', as_index=False).agg(months_observed=('month','nunique') if 'month' in e else ('property_id','size'), monthly_electricity_sum=('electricity_kwh','sum'), monthly_gas_sum=('gas_kwh','sum'))
        # Annualise observed monthly totals to 12 months only when fewer than 12 distinct months are present.
        if 'months_observed' in agg:
            factor=np.where(agg['months_observed']>0, 12/agg['months_observed'], np.nan)
            agg['annualised_electricity_from_monthly']=agg['monthly_electricity_sum']*factor
            agg['annualised_gas_from_monthly']=agg['monthly_gas_sum']*factor
        p=p.merge(agg[['property_id','months_observed','annualised_electricity_from_monthly','annualised_gas_from_monthly']],on='property_id',how='left')
        p['annual_electricity_kwh']=p['annualised_electricity_from_monthly'].combine_first(p.get('annual_electricity_kwh', pd.Series(index=p.index,dtype=float)))
        p['annual_gas_kwh']=p['annualised_gas_from_monthly'].combine_first(p.get('annual_gas_kwh', pd.Series(index=p.index,dtype=float)))
    for c, default in {'property_type':'unknown','age':0,'floor_area':np.nan,'bedrooms':np.nan,'energy_rating':'D','heating':'unknown','insulation':'unknown','glazing':'unknown','solar':False,'smart_controls':False,'region':'unknown','value_current_gbp':np.nan,'has_loft':True,'cavity_walls':False,'wall_insulated':False,'draught_proofed':False,'heat_pump_space':True,'solar_roof_suitable':True,'annual_electricity_kwh':np.nan,'annual_gas_kwh':np.nan,'space_heat_kwh':np.nan,'hot_water_kwh':np.nan,'base_electricity_kwh':np.nan,'other_gas_kwh':np.nan,'pv_offset_kwh':0}.items():
        if c not in p: p[c]=default
    p['energy_intensity_kwh_m2']=(p['annual_electricity_kwh']+p['annual_gas_kwh'])/p['floor_area'].replace(0,np.nan)
    if not labels.empty:
        keep=list(dict.fromkeys(c for c in TARGETS+['property_id','split','green_home_score','classification','package_ids','value_scenario_low_gbp','value_scenario_high_gbp'] if c in labels))
        # Keep reference label split authoritative where available.
        p=p.merge(labels[keep].drop(columns=['split'],errors='ignore'),on='property_id',how='left',suffixes=('','_label'))
    if 'split' not in p: p['split']='train'
    return p, labels, sales

def _make_pipeline(df):
    numeric=[c for c in NUMERIC+BOOLS if c in df.columns]
    categorical=[c for c in CATEGORICAL if c in df.columns]
    numeric_pipe=Pipeline([('imputer',SimpleImputer(strategy='median', add_indicator=True)),('scale',StandardScaler())])
    cat_pipe=Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))])
    prep=ColumnTransformer([('num',numeric_pipe,numeric),('cat',cat_pipe,categorical)],remainder='drop')
    model=RandomForestRegressor(n_estimators=180,min_samples_leaf=3,random_state=42,n_jobs=-1)
    return Pipeline([('preprocess',prep),('model',model)])

def train_and_evaluate(data_dir=DATA_DIR, model_dir=MODEL_DIR, report_dir=REPORT_DIR):
    model_dir,report_dir=Path(model_dir),Path(report_dir); model_dir.mkdir(parents=True,exist_ok=True); report_dir.mkdir(parents=True,exist_ok=True)
    df, labels, sales=load_data(data_dir)
    if labels.empty: raise ValueError('reference_labels.csv is required for supervised training; inference can still use deterministic scoring without labels.')
    # Use explicit property-level split field, never random row split.
    if 'split' not in df or not set(df['split'].dropna().str.lower()).intersection({'train'}):
        raise ValueError('Dataset must include property-level split values: train, validation, test.')
    train=df[df['split'].str.lower()=='train'].copy(); val=df[df['split'].str.lower().isin(['validation','val'])].copy(); test=df[df['split'].str.lower()=='test'].copy()
    if train.empty or test.empty: raise ValueError('Both train and test property splits must be non-empty.')
    metrics={'dataset':{'properties':int(len(df)),'train':int(len(train)),'validation':int(len(val)),'test':int(len(test)),'monthly_energy_rows':int(pd.read_csv(Path(data_dir)/'energy_monthly.csv').shape[0]) if (Path(data_dir)/'energy_monthly.csv').exists() else 0},'targets':{},'limitations':['Supplied reference labels are synthetic/reference estimates; held-out agreement does not prove real-world retrofit causality.','Property value impact is indicative, not a guaranteed causal uplift.']}
    feature_cols=[c for c in FEATURES if c in df.columns]
    for target in TARGETS:
        if target not in df.columns: continue
        fit_train=train[train[target].notna()]; fit_val=val[val[target].notna()]; fit_test=test[test[target].notna()]
        if len(fit_train)<10 or fit_test.empty: continue
        pipe=_make_pipeline(fit_train[feature_cols])
        pipe.fit(fit_train[feature_cols],fit_train[target])
        preds=pipe.predict(fit_test[feature_cols])
        metrics['targets'][target]={'train_rows':len(fit_train),'validation_rows':len(fit_val),'test_rows':len(fit_test),'test_mae':round(float(mean_absolute_error(fit_test[target],preds)),4),'test_rmse':round(float(np.sqrt(mean_squared_error(fit_test[target],preds))),4),'test_r2':round(float(r2_score(fit_test[target],preds)),4) if len(fit_test)>1 else None}
        if len(fit_val)>0:
            vp=pipe.predict(fit_val[feature_cols]); metrics['targets'][target]['validation_mae']=round(float(mean_absolute_error(fit_val[target],vp)),4)
        joblib.dump({'pipeline':pipe,'features':feature_cols,'target':target},model_dir/f'{target}.joblib')
    # Score classification is deterministic and must not be learned from arbitrary labels.
    metrics['green_home_score']='Deterministic weighted scoring engine; no ML training required.'
    with open(report_dir/'evaluation.json','w') as f: json.dump(metrics,f,indent=2)
    with open(model_dir/'metadata.json','w') as f: json.dump({'features':feature_cols,'targets':list(metrics['targets'].keys()),'model':'RandomForestRegressor','random_state':42,'split_policy':'property-level supplied split','scoring':'deterministic weighted scoring'},f,indent=2)
    return metrics

def load_models(model_dir=MODEL_DIR):
    result={}
    for f in Path(model_dir).glob('*.joblib'):
        try:
            item=joblib.load(f); result[item['target']]=item
        except Exception:
            continue
    return result

def predict_targets(p: dict, models=None):
    models=models if models is not None else load_models()
    df=pd.DataFrame([p])
    for col, default in [('annual_electricity_kwh', 0), ('annual_gas_kwh', 0), ('floor_area', 1)]:
        if col not in df.columns: df[col] = default
        df[col] = pd.to_numeric(df[col], errors='coerce')
    area=df['floor_area'].replace(0,np.nan)
    df['energy_intensity_kwh_m2']=(df['annual_electricity_kwh']+df['annual_gas_kwh'])/area
    out={}
    for target,item in models.items():
        cols=item['features']
        for c in cols:
            if c not in df: df[c]=np.nan
        try: out[target]=float(item['pipeline'].predict(df[cols])[0])
        except Exception: continue
    return out
