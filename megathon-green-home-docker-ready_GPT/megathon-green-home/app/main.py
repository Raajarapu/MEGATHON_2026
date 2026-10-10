from __future__ import annotations
import os, tempfile
from pathlib import Path
import pandas as pd
from flask import Flask, jsonify, render_template, request
from .engine import score_property, feasible_retrofits, simulate_package
from .ml import DATA_DIR, load_data, load_models, predict_targets, train_and_evaluate

ROOT=Path(__file__).resolve().parents[1]

def create_app():
    app=Flask(__name__, template_folder='templates', static_folder='static')
    app.config['MAX_CONTENT_LENGTH']=30*1024*1024
    app.config['SECRET_KEY']=os.getenv('SECRET_KEY','development-only-change-me')
    state={'uploaded_properties':None,'uploaded_energy':None}

    def get_data():
        p, labels, sales=load_data(DATA_DIR)
        if state['uploaded_properties'] is not None:
            up=state['uploaded_properties'].copy()
            if state['uploaded_energy'] is not None:
                en=state['uploaded_energy']
                agg=en.groupby('property_id',as_index=False).agg(months_observed=('month','nunique') if 'month' in en else ('property_id','size'),annual_e=('electricity_kwh','sum'),annual_g=('gas_kwh','sum'))
                if 'month' in en:
                    factor=12/agg['months_observed'].clip(lower=1)
                    agg['annual_e']*=factor; agg['annual_g']*=factor
                up=up.drop(columns=['annual_electricity_kwh','annual_gas_kwh'],errors='ignore').merge(agg[['property_id','annual_e','annual_g']],on='property_id',how='left').rename(columns={'annual_e':'annual_electricity_kwh','annual_g':'annual_gas_kwh'})
            p=up
        return p, labels, sales

    def find_property(pid):
        p, labels, sales=get_data()
        row=p[p['property_id'].astype(str)==str(pid)]
        if row.empty: return None
        return row.iloc[0].to_dict()

    @app.get('/')
    def home(): return render_template('index.html')

    @app.get('/api/health')
    def health():
        return jsonify({'status':'ok','service':'megathon-green-home','models_loaded':len(load_models()),'data_ready':(DATA_DIR/'properties.csv').exists()})

    @app.get('/api/summary')
    def summary():
        p,labels,sales=get_data()
        if p.empty: return jsonify({'error':'No property data loaded'}),400
        p=p.copy(); p['annual_electricity_kwh']=pd.to_numeric(p.get('annual_electricity_kwh',0),errors='coerce').fillna(0); p['annual_gas_kwh']=pd.to_numeric(p.get('annual_gas_kwh',0),errors='coerce').fillna(0)
        scores=[score_property(r) for r in p.to_dict('records')]
        classes=pd.Series([s['classification'] for s in scores]).value_counts().to_dict()
        return jsonify({'properties':int(len(p)),'mean_score':round(sum(s['score'] for s in scores)/len(scores),1),'classifications':classes,'annual_electricity_kwh':round(float(p['annual_electricity_kwh'].sum()),1),'annual_gas_kwh':round(float(p['annual_gas_kwh'].sum()),1),'models_loaded':len(load_models()),'data_mode':'uploaded dataset' if state['uploaded_properties'] is not None else 'bundled sample'})

    @app.get('/api/properties')
    def properties():
        p,_,_=get_data(); limit=min(max(int(request.args.get('limit',100)),1),1000)
        search=request.args.get('q','').strip().lower()
        if search: p=p[p['property_id'].astype(str).str.lower().str.contains(search,regex=False)]
        out=[]
        for r in p.head(limit).to_dict('records'):
            s=score_property(r); out.append({'property_id':str(r.get('property_id')),'property_type':r.get('property_type','unknown'),'region':r.get('region','unknown'),'energy_rating':r.get('energy_rating','unknown'),'floor_area':round(float(r.get('floor_area') or 0),1),'score':s['score'],'classification':s['classification']})
        return jsonify({'items':out,'count':len(out)})

    @app.get('/api/property/<property_id>')
    def property_detail(property_id):
        p=find_property(property_id)
        if p is None: return jsonify({'error':f'Property {property_id} not found'}),404
        score=score_property(p); retrofits=feasible_retrofits(p); scenario=simulate_package(p,[x['id'] for x in retrofits])
        preds=predict_targets(p)
        val=float(p.get('value_current_gbp') or 0)
        return jsonify({'property':{k:(v.item() if hasattr(v,'item') else v) for k,v in p.items() if not (isinstance(v,float) and pd.isna(v))},'score':score,'recommendations':retrofits,'scenario':scenario,'predictions':preds,'value_estimate':{'current_value_gbp':round(val,2) if val else None,'low_gbp':round(preds.get('value_scenario_low_gbp',preds.get('value_reference_central_gbp',val)*0.98),2) if preds.get('value_scenario_low_gbp',preds.get('value_reference_central_gbp',val)) else None,'high_gbp':round(preds.get('value_scenario_high_gbp',preds.get('value_reference_central_gbp',val)*1.02),2) if preds.get('value_scenario_high_gbp',preds.get('value_reference_central_gbp',val)) else None,'label':'Indicative model estimate; not a guaranteed causal retrofit uplift.'}})

    @app.post('/api/what-if')
    def what_if():
        body=request.get_json(silent=True) or {}; pid=body.get('property_id'); ids=body.get('retrofits',[])
        if not pid: return jsonify({'error':'property_id is required'}),400
        if not isinstance(ids,list): return jsonify({'error':'retrofits must be a list of retrofit IDs'}),400
        p=find_property(pid)
        if p is None: return jsonify({'error':f'Property {pid} not found'}),404
        allowed={r['id'] for r in feasible_retrofits(p)} | {'loft_insulation','cavity_wall_insulation','draught_proofing','double_glazing','smart_thermostat','heat_pump','solar_panels'}
        invalid=[x for x in ids if x not in allowed]
        if invalid: return jsonify({'error':'Unknown retrofit IDs','invalid':invalid}),400
        score=score_property(p); sim=simulate_package(p,ids)
        # The LLM is intentionally not asked to compute anything. UI explanation is templated from computed values.
        return jsonify({'property_id':pid,'selected_retrofits':ids,'score_before':score['score'],'scenario':sim,'explanation':f"This scenario estimates {sim['electricity_saving_kwh']:.0f} kWh/year electricity change and {sim['gas_saving_kwh']:.0f} kWh/year gas change. Estimated carbon reduction is {sim['carbon_reduction_kg_co2e']:.0f} kg CO₂e/year. These are modelled estimates, not guarantees."})

    @app.post('/api/upload-dataset')
    def upload_dataset():
        f=request.files.get('properties'); ef=request.files.get('energy_monthly')
        if not f: return jsonify({'error':'Upload properties CSV using form field "properties".'}),400
        try:
            up=pd.read_csv(f)
            required={'property_id','floor_area','energy_rating','heating','insulation','glazing'}
            missing=sorted(required-set(up.columns))
            if missing: return jsonify({'error':'Properties CSV is missing required columns.','missing_columns':missing}),400
            if not {'annual_electricity_kwh','annual_gas_kwh'}.issubset(up.columns) and ef is None:
                return jsonify({'error':'Provide annual electricity/gas columns or upload energy_monthly CSV too.'}),400
            if up['property_id'].isna().any(): return jsonify({'error':'property_id contains missing values.'}),400
            if up['property_id'].duplicated().any(): return jsonify({'error':'properties.csv must have one row per property_id.'}),400
            energy=None
            if ef:
                energy=pd.read_csv(ef)
                req={'property_id','electricity_kwh','gas_kwh'}
                if not req.issubset(energy.columns): return jsonify({'error':'energy_monthly CSV must contain property_id, electricity_kwh, gas_kwh.'}),400
            state['uploaded_properties']=up; state['uploaded_energy']=energy
            return jsonify({'status':'loaded_for_inference','properties':len(up),'energy_rows':len(energy) if energy is not None else 0,'columns':list(up.columns),'note':'This upload is used for scoring/recommendation preview only. It does not silently retrain models or treat unverified labels as truth.'})
        except Exception as exc: return jsonify({'error':f'Could not read CSV: {exc}'}),400

    @app.post('/api/retrain')
    def retrain():
        try:
            metrics=train_and_evaluate(); return jsonify({'status':'trained','evaluation':metrics})
        except Exception as exc: return jsonify({'error':str(exc)}),400

    return app

app=create_app()
