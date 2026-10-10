from app.engine import score_property, simulate_package, classify

def test_score_weighting_and_range():
    p={'energy_rating':'A','floor_area':100,'annual_electricity_kwh':2000,'annual_gas_kwh':3000,'insulation':'excellent','heating':'heat_pump','glazing':'triple','solar':True,'smart_controls':True}
    s=score_property(p)
    assert 0 <= s['score'] <= 100
    assert sum(v['weight'] for v in s['components'].values()) == 1.0
    assert s['classification'] == classify(s['score'])

def test_solar_does_not_raise_grid_imports():
    p={'floor_area':100,'annual_electricity_kwh':4000,'annual_gas_kwh':5000,'space_heat_kwh':3000,'base_electricity_kwh':4000,'other_gas_kwh':2000}
    base=simulate_package(p,[]); solar=simulate_package(p,['solar_panels'])
    assert solar['after_electricity_kwh'] <= base['after_electricity_kwh']

def test_heat_pump_can_increase_electricity_and_reduce_gas():
    p={'floor_area':100,'annual_electricity_kwh':3000,'annual_gas_kwh':10000,'space_heat_kwh':7000,'base_electricity_kwh':3000,'other_gas_kwh':3000}
    out=simulate_package(p,['heat_pump'])
    assert out['after_electricity_kwh'] > out['baseline_electricity_kwh']
    assert out['after_gas_kwh'] < out['baseline_gas_kwh']
