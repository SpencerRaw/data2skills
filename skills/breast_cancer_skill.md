# breast_cancer_diagnostic_optimized
Domain: medical_diagnostics  
Features: mean radius, mean texture, mean perimeter, mean area, mean smoothness, mean compactness, mean concavity, mean concave points, mean symmetry, mean fractal dimension, radius error, texture error, perimeter error, area error, smoothness error, compactness error, concavity error, concave points error, symmetry error, fractal dimension error, worst radius, worst texture, worst perimeter, worst area, worst smoothness, worst compactness, worst concavity, worst concave points, worst symmetry, worst fractal dimension  
Target: target  

## Performance
- **accuracy**: 0.8901098901098901
- **n_rules**: 9
- **epochs_trained**: 0

## Rules
### Rule R1
- **IF** worst concave points > 0.18
- **THEN** predict `malignant`
- **Confidence**: 0.60
- **Support**: 138/138 (100.0%)

### Rule R2
- **IF** mean concave points > 0.09
- **THEN** predict `malignant`
- **Confidence**: 0.60
- **Support**: 138/138 (100.0%)

### Rule R3
- **IF** worst concave points < 0.07
- **THEN** predict `benign`
- **Confidence**: 0.60
- **Support**: 226/226 (100.0%)

### Rule R4
- **IF** mean concave points < 0.03
- **THEN** predict `benign`
- **Confidence**: 0.60
- **Support**: 226/226 (100.0%)

### Rule R5
- **IF** mean radius > 15.758
- **THEN** predict `malignant`
- **Confidence**: 0.50

### Rule R6
- **IF** mean texture > 19.918
- **THEN** predict `malignant`
- **Confidence**: 0.50

### Rule R7
- **IF** mean perimeter > 102.945
- **THEN** predict `malignant`
- **Confidence**: 0.50

### Rule R8
- **IF** mean area > 630.333
- **THEN** predict `benign`
- **Confidence**: 0.50

### Rule R9
- **IF** mean smoothness > 0.092
- **THEN** predict `benign`
- **Confidence**: 0.50
