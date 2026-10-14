# [23] app_core.py: core logic of the app (load model, validate input, predict, severity, advice)
import joblib
import pandas as pd

ADVICE = {
    'Healthy': 'القيم في المعدل الطبيعي. حافظ على غذاء متوازن.',
    'Iron deficiency anemia': 'غالباً نقص حديد. الأطعمة الغنية بالحديد (لحوم، كبدة، عدس، سبانخ) مع مصدر فيتامين C بتساعد. ما تاخدش مكملات حديد قبل استشارة الطبيب.',
    'Macrocytic anemia': 'غالباً نقص B12 أو فولات. الأطعمة الغنية بيهم (لحوم، بيض، ألبان، ورقيات) مهمة. الطبيب هو اللي يحدد السبب والعلاج.',
    'Normocytic hypochromic anemia': 'ممكن يكون نقص حديد في مرحلة بدري أو مرض مزمن. محتاج تحاليل إضافية (فيريتين وحديد) وفحص طبي.',
    'Normocytic normochromic anemia': 'ممكن يكون مرتبط بأمراض مزمنة أو فقدان دم. محتاج تقييم طبي لتحديد السبب.',
    'Other microcytic anemia': 'كرات الدم صغيرة. ممكن يكون ثلاسيميا أو نقص حديد. محتاج تحاليل إضافية قبل أي مكملات.',
}
DISCLAIMER = 'للتثقيف فقط، وليس بديلاً عن استشارة الطبيب.'
URGENT = 'الهيموجلوبين منخفض جداً. توجّه إلى الطبيب في أقرب وقت.'
# [25a] app_core.py: hard limits = physically impossible values (rejected, not just warned)
HARD_LIMITS = {
    'WBC': (0.5, 100), 'RBC': (1, 10), 'HGB': (3, 25), 'HCT': (10, 70),
    'MCV': (40, 150), 'MCH': (10, 50), 'MCHC': (20, 45), 'PLT': (5, 1500),
}

def load_artifacts(path):
    return joblib.load(path)  # dict: model, features, ranges


def get_severity(hgb, sex):
    normal_limit = 13.0 if sex == 'male' else 12.0
    if hgb >= normal_limit:
        return 'No anemia'
    elif hgb >= 11.0:
        return 'Mild'
    elif hgb >= 8.0:
        return 'Moderate'
    else:
        return 'Severe'


def check_ranges(values, features, ranges):
    warnings = []
    for name in features:
        low, high = ranges.loc['min', name], ranges.loc['max', name]
        if not (low <= values[name] <= high):
            warnings.append(f"{name}={values[name]} is outside the training range ({low} to {high}); check the unit")
    return warnings


def analyze(values, sex, artifacts):
    features = artifacts['features']
    missing = [f for f in features if f not in values]
    if missing:
        return {'error': f"Missing values: {missing}"}
    if sex not in ('male', 'female'):
        return {'error': "sex must be 'male' or 'female'"}
            # [25b] app_core.py: inside analyze(), reject impossible values before predicting
    impossible = []
    for name in features:
        low, high = HARD_LIMITS[name]
        if not (low <= values[name] <= high):
            impossible.append(f"{name}={values[name]} is outside the possible range ({low} to {high}); check the unit")
    if impossible:
        return {'error': impossible}
    row = pd.DataFrame([values])[features]
    model = artifacts['model']
    diagnosis = model.predict(row)[0]
    confidence = model.predict_proba(row)[0].max()
    severity = get_severity(values['HGB'], sex)

    return {
        'diagnosis': diagnosis,
        'confidence': round(float(confidence), 2),
        'severity': severity,
        'advice': ADVICE[diagnosis],
        'urgent': URGENT if severity == 'Severe' else None,
        'warnings': check_ranges(values, features, artifacts['ranges']),
        'disclaimer': DISCLAIMER,
    }


if __name__ == "__main__":
    artifacts = load_artifacts('models/anemia_model_8f_clean.joblib')
    test = {'WBC': 7.0, 'RBC': 4.8, 'HGB': 14.0, 'HCT': 42.0,
            'MCV': 88.0, 'MCH': 29.0, 'MCHC': 33.0, 'PLT': 250.0}
    print(analyze(test, 'male', artifacts))