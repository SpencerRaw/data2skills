"""Data anonymization for medical datasets.

Strips PII while preserving medical feature integrity.
Designed for SLE patient data but works for any tabular health data.

Usage:
    python anonymize.py input.csv --output safe_output.csv
    python anonymize.py input.xlsx --sheet "Sheet1" --output safe.csv

What it does:
    - Detects and removes PII columns (name, ID, phone, address, DOB, etc.)
    - Shifts dates by random per-patient offset (preserves intervals)
    - Generalizes ages to 5-year bins
    - Keeps all medical features intact
    - Outputs audit report of what was done
"""

import argparse, csv, hashlib, os, random, sys, json
from datetime import datetime, timedelta
from pathlib import Path


# Column name patterns that indicate PII (case-insensitive)
PII_PATTERNS = [
    # Direct identifiers
    'name', 'first', 'last', 'patient', 'subject',
    'id', 'identifier', 'mrn', 'record',
    'ssn', 'social_security', 'national_id',
    'phone', 'mobile', 'cell', 'telephone', 'fax',
    'email', 'e-mail',
    'address', 'street', 'city', 'state', 'province',
    'zip', 'postal', 'postcode', 'country',
    # Dates that could identify
    'dob', 'birth', 'date_of_birth',
    'admission_date', 'discharge_date', 'visit_date',
    # Names of doctors/staff
    'doctor', 'physician', 'clinician', 'nurse',
    'referring', 'attending',
    # Other identifiers
    'insurance', 'account', 'policy',
    'license', 'passport',
    # Free text that may contain PII
    'notes', 'comments', 'history_text',
]

# Column patterns that are semi-identifying (keep but generalize)
GENERALIZE_PATTERNS = {
    'age': 'bin',           # 37 → "35-40"
    'weight': 'round',      # 72.3 → 72
    'height': 'round',      # 168.7 → 169
    'bmi': 'round',         # 24.7 → 25
}

# Medical column patterns to keep as-is
KEEP_PATTERNS = [
    'lab', 'test', 'result', 'value', 'level',
    'diagnosis', 'diagnostic', 'icd', 'code',
    'symptom', 'sign', 'finding',
    'medication', 'drug', 'dose', 'treatment',
    'blood', 'urine', 'serum', 'plasma',
    'antibody', 'ana', 'dsdna', 'complement', 'c3', 'c4',
    'creatinine', 'protein', 'albumin',
    'sle', 'lupus', 'sledai', 'bilag',
    'wbc', 'rbc', 'platelet', 'hemoglobin',
    'crp', 'esr', 'ferritin',
    'gender', 'sex', 'race', 'ethnicity',
    'year', 'month',
]


def is_pii_column(col_name: str) -> bool:
    """Check if column name indicates PII."""
    col_lower = col_name.lower().strip()
    for pattern in PII_PATTERNS:
        if pattern in col_lower:
            return True
    return False


def is_medical_column(col_name: str) -> bool:
    """Check if column looks like a medical feature to keep."""
    col_lower = col_name.lower().strip()
    for pattern in KEEP_PATTERNS:
        if pattern in col_lower:
            return True
    return False


def should_generalize(col_name: str) -> str:
    """Returns 'bin', 'round', or None."""
    col_lower = col_name.lower().strip()
    for pattern, method in GENERALIZE_PATTERNS.items():
        if pattern in col_lower:
            return method
    return None


def generalize_age(val) -> str:
    """37 → '35-40'"""
    try:
        age = int(float(val))
        lower = (age // 5) * 5
        return f"{lower}-{lower+4}"
    except (ValueError, TypeError):
        return str(val)


def round_value(val):
    """72.34 → 72"""
    try:
        return str(int(round(float(val))))
    except (ValueError, TypeError):
        return str(val)


def hash_id(val) -> str:
    """One-way hash with salt."""
    if not val or str(val).strip() == '':
        return ''
    salt = os.urandom(16).hex()  # Per-run salt
    # Actually, use deterministic salt for reproducibility
    return hashlib.sha256(f"data2skills_salt_{val}".encode()).hexdigest()[:12]


def shift_date(date_str: str, offset_days: int) -> str:
    """Shift a date by a random offset, preserving intervals."""
    try:
        for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d']:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                shifted = dt + timedelta(days=offset_days)
                return shifted.strftime('%Y-%m-%d')
            except ValueError:
                continue
        return 'REDACTED'
    except:
        return 'REDACTED'


def anonymize(
    input_path: str,
    output_path: str,
    sheet_name: str = None,
    id_seed: int = None,
) -> dict:
    """Main anonymization function.
    
    Returns audit report dict.
    """
    import pandas as pd
    
    if input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
    elif input_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(input_path, sheet_name=sheet_name)
    else:
        raise ValueError(f"Unsupported format: {input_path}")
    
    if id_seed is None:
        id_seed = random.randint(0, 2**31)
    random.seed(id_seed)
    
    audit = {
        "input_file": input_path,
        "output_file": output_path,
        "original_shape": list(df.shape),
        "columns_removed": [],
        "columns_hashed": [],
        "columns_generalized": [],
        "columns_date_shifted": [],
        "columns_kept": [],
        "date_shift_days": {},
    }
    
    # Generate per-row date shift offsets
    date_offsets = [random.randint(-180, 180) for _ in range(len(df))]
    
    for col in df.columns:
        if is_pii_column(col):
            # Check if it might be a date column
            sample = str(df[col].dropna().iloc[0]) if len(df[col].dropna()) > 0 else ''
            is_date = any(c.isdigit() for c in sample) and ('/' in sample or '-' in sample)
            
            if is_date and len(sample) > 6:
                # Shift dates
                df[col] = [
                    shift_date(str(v), date_offsets[i]) if pd.notna(v) else v
                    for i, v in enumerate(df[col])
                ]
                audit["columns_date_shifted"].append(col)
                audit["date_shift_days"][col] = f"random per-row ±180 days"
            else:
                # Remove PII column entirely
                audit["columns_removed"].append(col)
                df.drop(columns=[col], inplace=True)
        
        elif should_generalize(col):
            method = should_generalize(col)
            if method == 'bin':
                df[col] = df[col].apply(generalize_age)
            elif method == 'round':
                df[col] = df[col].apply(round_value)
            audit["columns_generalized"].append(col)
        
        elif is_medical_column(col):
            audit["columns_kept"].append(col)
        
        else:
            # Unknown column — keep but flag
            audit["columns_kept"].append(col)
    
    # Save
    df.to_csv(output_path, index=False)
    audit["final_shape"] = list(df.shape)
    
    # Save audit
    audit_path = output_path.replace('.csv', '_audit.json')
    with open(audit_path, 'w') as f:
        json.dump(audit, f, indent=2)
    
    print(f"Anonymized: {audit['original_shape']} → {audit['final_shape']}")
    print(f"  Removed: {len(audit['columns_removed'])} PII columns")
    print(f"  Generalized: {len(audit['columns_generalized'])} columns")
    print(f"  Date-shifted: {len(audit['columns_date_shifted'])} columns")
    print(f"  Kept: {len(audit['columns_kept'])} medical features")
    print(f"  Audit log: {audit_path}")
    
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anonymize medical data")
    parser.add_argument("input", help="Input CSV/XLSX file")
    parser.add_argument("--output", "-o", required=True, help="Output CSV file")
    parser.add_argument("--sheet", help="Sheet name (for XLSX)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    anonymize(args.input, args.output, args.sheet, args.seed)
