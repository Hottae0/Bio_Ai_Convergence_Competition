<img width="640" height="370" alt="image" src="https://github.com/user-attachments/assets/a623a671-438f-4f02-bd5c-36ac740114df" />


# 분자 속성 예측 프로젝트

이 프로젝트는 SMILES 문자열로 표현된 분자 구조로부터 특정 속성 값을 예측하는 머신러닝 모델을 학습하고(`train.py`), 학습된 모델을 사용해 새로운 분자를 예측(`predict.py`)하는 파이썬 스크립트를 포함하고 있습니다.

## 주요 기능

-   **피처 엔지니어링**: RDKit 라이브러리를 사용하여 각 SMILES 문자열로부터 다음과 같은 분자 피처를 생성합니다.
    -   **Morgan Fingerprints**: 분자의 구조적 특징을 나타내는 원형 지문 (2048 비트)
    -   **MACCS Keys**: 167개의 구조적 키로 구성된 서브구조 지문
    -   **2D Descriptors**: 분자량, 로그P(logP) 등 200여 개의 2차원 물리화학적 특성
-   **모델 학습 및 예측**: XGBoost 모델을 사용하여 피처와 타겟 값의 관계를 학습하고, 새로운 데이터에 대한 예측을 수행합니다.

---

## 📄 파일 구조

-   `train.py`: 모델을 학습시키고 `model.pkl` 파일을 생성합니다.
-   `predict.py`: 학습된 `model.pkl`을 이용해 새로운 분자를 예측합니다.
-   `model.pkl`: `train.py` 실행 후 생성되는 최종 모델 파일입니다.
-   `submission.csv`: `predict.py` 실행 후 생성되는 최종 예측 결과 파일입니다.

---

## ⚙️ 설치 및 환경 설정

스크립트를 실행하기 전에 필요한 라이브러리를 설치해야 합니다.

```bash
pip install pandas numpy scikit-learn xgboost rdkit-pypi

