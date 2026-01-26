"""
Generate synthetic student admission dataset with built-in biases for fairness testing.

Usage: python generate_dataset.py

This script creates a reproducible dataset with RANDOM_SEED=42.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Configuration
RANDOM_SEED = 42
N_SAMPLES = 12000

def generate_dataset():
    """Generate complete synthetic student admission dataset."""
    np.random.seed(RANDOM_SEED)
    
    print(f"Generating {N_SAMPLES:,} synthetic student applications...")
    print(f"Random seed: {RANDOM_SEED}\n")
    
    # === PROTECTED ATTRIBUTES ===
    
    gender = np.random.choice(['Male', 'Female', 'Non-binary'], 
                              size=N_SAMPLES, p=[0.48, 0.48, 0.04])
    
    race = np.random.choice(['White', 'Black', 'Hispanic', 'Asian', 'Native American', 'Other'],
                            size=N_SAMPLES, p=[0.55, 0.13, 0.18, 0.10, 0.02, 0.02])
    
    # SES correlated with race (systemic inequality)
    socioeconomic_status = []
    for r in race:
        if r == 'White':
            ses = np.random.choice(['Low', 'Middle', 'High'], p=[0.25, 0.45, 0.30])
        elif r == 'Asian':
            ses = np.random.choice(['Low', 'Middle', 'High'], p=[0.20, 0.40, 0.40])
        elif r in ['Black', 'Hispanic', 'Native American']:
            ses = np.random.choice(['Low', 'Middle', 'High'], p=[0.45, 0.40, 0.15])
        else:
            ses = np.random.choice(['Low', 'Middle', 'High'], p=[0.33, 0.34, 0.33])
        socioeconomic_status.append(ses)
    socioeconomic_status = np.array(socioeconomic_status)
    
    # First generation correlated with SES
    first_generation = []
    for ses in socioeconomic_status:
        if ses == 'Low':
            fg = np.random.choice(['Yes', 'No'], p=[0.65, 0.35])
        elif ses == 'Middle':
            fg = np.random.choice(['Yes', 'No'], p=[0.35, 0.65])
        else:
            fg = np.random.choice(['Yes', 'No'], p=[0.10, 0.90])
        first_generation.append(fg)
    first_generation = np.array(first_generation)
    
    region = np.random.choice(['Northeast', 'Southeast', 'Midwest', 'Southwest', 'West'],
                             size=N_SAMPLES, p=[0.22, 0.20, 0.18, 0.18, 0.22])
    
    urban_rural = []
    for ses in socioeconomic_status:
        if ses == 'High':
            ur = np.random.choice(['Urban', 'Suburban', 'Rural'], p=[0.35, 0.55, 0.10])
        elif ses == 'Middle':
            ur = np.random.choice(['Urban', 'Suburban', 'Rural'], p=[0.30, 0.50, 0.20])
        else:
            ur = np.random.choice(['Urban', 'Suburban', 'Rural'], p=[0.40, 0.30, 0.30])
        urban_rural.append(ur)
    urban_rural = np.array(urban_rural)
    
    disability_status = np.random.choice(['Yes', 'No'], size=N_SAMPLES, p=[0.12, 0.88])
    
    print("✓ Generated protected attributes")
    
    # === ACADEMIC FEATURES ===
    
    def generate_gpa(ses, race):
        base_mean = 3.2
        mean_adj = 0.25 if ses == 'High' else (0.10 if ses == 'Middle' else -0.10)
        race_adj = 0.08 if race in ['Asian', 'White'] else -0.05
        gpa = np.clip(np.random.normal(base_mean + mean_adj + race_adj, 0.4), 0.0, 4.0)
        return round(gpa, 2)
    
    def generate_sat(ses, race, urban_rural):
        base_mean = 1050
        mean_adj = 150 if ses == 'High' else (50 if ses == 'Middle' else -80)
        race_adj = 80 if race == 'Asian' else (40 if race == 'White' else -30)
        ur_adj = 30 if urban_rural == 'Suburban' else (10 if urban_rural == 'Urban' else -20)
        sat = np.clip(np.random.normal(base_mean + mean_adj + race_adj + ur_adj, 150), 400, 1600)
        return int(round(sat / 10) * 10)
    
    def generate_act(sat_score):
        act = (sat_score - 400) / 1200 * 35 + 1 + np.random.normal(0, 2)
        return int(np.clip(act, 1, 36))
    
    gpa = np.array([generate_gpa(ses, r) for ses, r in zip(socioeconomic_status, race)])
    sat_score = np.array([generate_sat(ses, r, ur) for ses, r, ur in zip(socioeconomic_status, race, urban_rural)])
    act_score = np.array([generate_act(sat) for sat in sat_score])
    
    # AP courses (wealth dependent)
    ap_courses = []
    for ses in socioeconomic_status:
        if ses == 'High':
            ap = int(np.clip(np.random.normal(6, 2), 0, 12))
        elif ses == 'Middle':
            ap = int(np.clip(np.random.normal(3, 2), 0, 12))
        else:
            ap = int(np.clip(np.random.normal(1, 1.5), 0, 12))
        ap_courses.append(ap)
    ap_courses = np.array(ap_courses)
    
    honors_courses = np.array([int(np.clip(np.random.normal(ap * 1.2 + 2, 2), 0, 15)) for ap in ap_courses])
    class_rank_percentile = np.array([int(np.clip((g / 4.0) * 100 + np.random.normal(0, 10), 0, 100)) for g in gpa])
    
    # Extracurriculars (wealth dependent)
    extracurriculars = []
    for ses in socioeconomic_status:
        if ses == 'High':
            extra = int(np.clip(np.random.normal(5, 2), 0, 10))
        elif ses == 'Middle':
            extra = int(np.clip(np.random.normal(3, 1.5), 0, 10))
        else:
            extra = int(np.clip(np.random.normal(1.5, 1), 0, 10))
        extracurriculars.append(extra)
    extracurriculars = np.array(extracurriculars)
    
    leadership_positions = np.array([int(np.clip(np.random.poisson(ex / 2.5), 0, 5)) for ex in extracurriculars])
    community_service_hours = np.array([int(np.clip(np.random.gamma(2, 30), 0, 500)) for _ in range(N_SAMPLES)])
    awards_honors = np.array([int(np.clip(np.random.poisson(max(0, (g - 2.5) * 2)), 0, 8)) for g in gpa])
    
    # Essay score (slight gender bias)
    essay_score = []
    for g in gender:
        score = np.clip(np.random.normal(6.5 if g == 'Female' else 6.0, 1.5), 1, 10)
        essay_score.append(round(score, 1))
    essay_score = np.array(essay_score)
    
    # Recommendation strength (SES advantage)
    recommendation_strength = []
    for rank, ses in zip(class_rank_percentile, socioeconomic_status):
        base = rank / 100 * 8 + 2
        if ses == 'High':
            base += 0.5
        rec = np.clip(np.random.normal(base, 1), 1, 10)
        recommendation_strength.append(round(rec, 1))
    recommendation_strength = np.array(recommendation_strength)
    
    print("✓ Generated academic features")
    
    # === ADMISSION DECISIONS (WITH BIASES) ===
    
    def calculate_admission_probability(idx):
        # Academic merit
        academic_score = (
            (gpa[idx] / 4.0) * 0.25 +
            (sat_score[idx] / 1600) * 0.20 +
            (act_score[idx] / 36) * 0.15 +
            (ap_courses[idx] / 12) * 0.10 +
            (class_rank_percentile[idx] / 100) * 0.15 +
            (extracurriculars[idx] / 10) * 0.05 +
            (essay_score[idx] / 10) * 0.05 +
            (recommendation_strength[idx] / 10) * 0.05
        )
        
        prob = academic_score
        
        # Built-in biases
        if gender[idx] == 'Male':
            prob += 0.02
        elif gender[idx] == 'Female':
            prob += 0.01
        
        if race[idx] in ['White', 'Asian']:
            prob += 0.03
        elif race[idx] in ['Black', 'Hispanic', 'Native American']:
            prob -= 0.02
        
        if socioeconomic_status[idx] == 'High':
            prob += 0.04
        elif socioeconomic_status[idx] == 'Low':
            prob -= 0.02
        
        if region[idx] in ['Northeast', 'West']:
            prob += 0.01
        
        if first_generation[idx] == 'Yes':
            prob -= 0.01
        
        prob += np.random.normal(0, 0.05)
        return np.clip(prob, 0, 1)
    
    admission_probabilities = np.array([calculate_admission_probability(i) for i in range(N_SAMPLES)])
    threshold = np.percentile(admission_probabilities, 70)
    admitted = (admission_probabilities > threshold).astype(int)
    
    print(f"✓ Generated admission decisions (rate: {admitted.mean():.1%})")
    
    # === CREATE DATAFRAME ===
    
    df = pd.DataFrame({
        'gender': gender,
        'race': race,
        'socioeconomic_status': socioeconomic_status,
        'first_generation': first_generation,
        'region': region,
        'urban_rural': urban_rural,
        'disability_status': disability_status,
        'gpa': gpa,
        'sat_score': sat_score,
        'act_score': act_score,
        'ap_courses': ap_courses,
        'honors_courses': honors_courses,
        'class_rank_percentile': class_rank_percentile,
        'extracurriculars': extracurriculars,
        'leadership_positions': leadership_positions,
        'community_service_hours': community_service_hours,
        'awards_honors': awards_honors,
        'essay_score': essay_score,
        'recommendation_strength': recommendation_strength,
        'admitted': admitted
    })
    
    return df

def save_datasets(df, output_dir):
    """Save full, train, and test datasets."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Full dataset
    full_path = output_dir / 'student_admission_full.csv'
    df.to_csv(full_path, index=False)
    print(f"\n✓ Saved full dataset: {full_path.relative_to(Path.cwd().parent.parent)}")
    print(f"  Rows: {len(df):,}, Columns: {len(df.columns)}")
    
    # Train/test split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED, stratify=df['admitted'])
    
    train_path = output_dir / 'train.csv'
    test_path = output_dir / 'test.csv'
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"✓ Saved train set: {train_path.relative_to(Path.cwd().parent.parent)}")
    print(f"  Rows: {len(train_df):,}, Admission rate: {train_df['admitted'].mean():.1%}")
    print(f"✓ Saved test set: {test_path.relative_to(Path.cwd().parent.parent)}")
    print(f"  Rows: {len(test_df):,}, Admission rate: {test_df['admitted'].mean():.1%}")

def main():
    print("="*60)
    print("Student Admission Dataset Generator")
    print("="*60)
    
    df = generate_dataset()
    
    output_dir = Path(__file__).parent / '../../resources/datasets/student-admission'
    save_datasets(df, output_dir)
    
    print("\n" + "="*60)
    print("Dataset generation complete!")
    print("="*60)

if __name__ == '__main__':
    main()
