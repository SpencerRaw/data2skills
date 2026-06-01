# SLE Phenotype Discovery
Domain: medical_phenotyping
Patients: 1049 | Features: 135
Phenotypes discovered: 4

## Global Insights
- Most variable features: 新出现脑血管意外（有-1  无-0）（8分）, 超敏肌钙蛋白I(CTnI，ng/mL), 股骨头坏死（有-1  无-0） (CV: 2288.0%, 2043.4%, 1616.3%)
- Strongest correlation: 病程 XX天 ↔ 柳氮磺胺嘧啶（未使用0 使用1） (r=nan)

## Discovered Phenotypes
## Phenotype 1 (n=542, 52% of cohort)

## Phenotype 3: 发热（有-1  无-0）（1分）-dominant (n=253, 24% of cohort)

**Elevated**:
  - 发热（有-1  无-0）（1分）: 0.5 vs cohort avg 0.2 (2.3x)
  - 血清免疫球蛋白 IgG（g/L): 20.4 vs cohort avg 15.7 (1.3x)
  - 血沉  mm/h: 58.6 vs cohort avg 38.7 (1.5x)
  - SLEDAI评分: 12.2 vs cohort avg 9.0 (1.4x)

**Reduced**:
  - 血红蛋白 g/L: 90.6 vs cohort avg 108.1
  - 红细胞计数  10^12/L: 3.2 vs cohort avg 3.8
  - 血高密度脂蛋白胆固醇（HDL-C）(mmol/L): 0.9 vs cohort avg 1.2
  - 血清补体C3  g/L（<0.9-2分）: 0.6 vs cohort avg 0.7
  - 舒缩压（mmHg） : 68.7 vs cohort avg 75.7

## Phenotype 4: 颈部血管斑块（有-1 无-0）-dominant (n=131, 12% of cohort)

**Elevated**:
  - 颈部血管斑块（有-1 无-0）: 0.3 vs cohort avg 0.0 (6.7x)
  - 颈部血管内膜增厚（有-1 无-0）: 0.3 vs cohort avg 0.0 (6.4x)
  - 骨质疏松 （有-1  无-0）: 0.3 vs cohort avg 0.0 (5.6x)
  - 是否患有高血压病（有-1  无-0）: 0.5 vs cohort avg 0.1 (3.4x)
  - 心脏超声异常（有-1  无-0）: 0.5 vs cohort avg 0.2 (2.7x)

## Phenotype 2: 尿蛋白定性 阴性-0 +-1 ++-2 +++3 ++++-4 （4分）-dominant (n=123, 12% of cohort)

**Elevated**:
  - 尿蛋白定性 阴性-0 +-1 ++-2 +++3 ++++-4 （4分）: 2.2 vs cohort avg 0.6 (3.7x)
  - 水肿（有-1  无-0）: 0.7 vs cohort avg 0.2 (4.4x)
  - 24小时总尿蛋白（mg）（>500--4分）: 3406.6 vs cohort avg 683.7 (5.0x)
  - 尿潜血定性 阴性-0 +-1 ++-2 +++3 ++++-4（4分）: 2.2 vs cohort avg 0.7 (3.0x)
  - 总胆固醇  mmol/L: 5.8 vs cohort avg 4.3 (1.3x)

**Reduced**:
  - 白蛋白（g/L）: 23.9 vs cohort avg 36.8
  - 总蛋白（g/L）: 50.2 vs cohort avg 67.0


## Feature Associations
- **总胆固醇  mmol/L** ↑ → **血低密度脂蛋白胆固醇（LDL-C）(mmol/L)** ↑  (lift=2.57)
- **游离三碘甲状腺原氨酸 Pmol/L** ↑ → **游离甲状腺素 Pmol/L** ↑  (lift=2.31)
- **尿蛋白定性 阴性-0 +-1 ++-2 +++3 ** ↑ → **24小时总尿蛋白（mg）（>500--4分）** ↑  (lift=2.20)
- **体重（Kg）** ↑ → **BMI（Kg/M^2）** ↑  (lift=2.10)
- **尿蛋白定性 阴性-0 +-1 ++-2 +++3 ** ↑ → **尿潜血定性 阴性-0 +-1 ++-2 +++3 ** ↑  (lift=2.00)
- **总胆固醇  mmol/L** ↑ → **血高密度脂蛋白胆固醇（HDL-C）(mmol/L)** ↑  (lift=1.92)
- **尿潜血定性 阴性-0 +-1 ++-2 +++3 ** ↑ → **SLEDAI评分** ↑  (lift=1.88)
- **尿潜血定性 阴性-0 +-1 ++-2 +++3 ** ↑ → **24小时总尿蛋白（mg）（>500--4分）** ↑  (lift=1.84)
- **甘油三酯（TG, mmol/L）** ↑ → **血低密度脂蛋白胆固醇（LDL-C）(mmol/L)** ↑  (lift=1.78)
- **白细胞计数（10^9/L）（<3-1分）** ↑ → **中性粒细胞计数 10^9/L** ↑  (lift=1.76)
- **甘油三酯（TG, mmol/L）** ↑ → **总胆固醇  mmol/L** ↑  (lift=1.76)
- **尿蛋白定性 阴性-0 +-1 ++-2 +++3 ** ↑ → **SLEDAI评分** ↑  (lift=1.73)
- **血高密度脂蛋白胆固醇（HDL-C）(mmol/L)** ↑ → **血低密度脂蛋白胆固醇（LDL-C）(mmol/L)** ↑  (lift=1.73)
- **凝血酶原时间(Sec)** ↑ → **活化部分凝血活酶时间(Sec)** ↑  (lift=1.73)
- **血清补体C3  g/L（<0.9-2分）** ↑ → **血清补体C4   g/L（<0.1-2分）** ↑  (lift=1.73)
