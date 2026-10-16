# [24] test_app.py: manual tests of analyze() for normal, low-HGB, wrong-unit and missing-value cases
from app_core import load_artifacts, analyze

artifacts = load_artifacts('models/anemia_model_8f_clean.joblib')

base = {'WBC': 7.0, 'RBC': 4.8, 'HGB': 14.0, 'HCT': 42.0,
        'MCV': 88.0, 'MCH': 29.0, 'MCHC': 33.0, 'PLT': 250.0}

low_hgb = dict(base, HGB=6.5, HCT=21.0, MCV=68.0, MCH=20.0, MCHC=29.0)
wrong_unit = dict(base, HGB=140.0)
missing = {k: v for k, v in base.items() if k != 'PLT'}

cases = [('normal', base), ('low hgb', low_hgb), ('wrong unit', wrong_unit), ('missing PLT', missing)]

for name, case in cases:
    result = analyze(case, 'female', artifacts)
    print(name, '->', result.get('diagnosis'), result.get('severity'),
          result.get('urgent') is not None, result.get('warnings'), result.get('errors'))