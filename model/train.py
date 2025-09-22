import pandas as pd
import numpy as np
import joblib
import os
import argparse
import warnings

# RDKit & 머신러닝 라이브러리
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, DataStructs, MACCSkeys
import xgboost as xgb

# 평가 지표
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

# --- Feature Engineering (predict.py와 동일) ---
N_BITS_MORGAN = 2048

# smile에서 필요한 거로 바꾸는 함수들
def smiles_to_morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = N_BITS_MORGAN) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return np.zeros((n_bits,), dtype=int)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def smiles_to_maccs_keys(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return np.zeros((167,), dtype=int)
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros((167,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


MANUAL_3D_DESC_NAMES = {
    'Asphericity', 'Eccentricity', 'InertialShapeFactor', 'NPR1', 'NPR2',
    'PMI1', 'PMI2', 'PMI3', 'RadiusOfGyration', 'SpherocityIndex'
}
DESC_LIST_2D = [d[0] for d in Descriptors.descList if d[0] not in MANUAL_3D_DESC_NAMES]


def smiles_to_rdkit_descriptors_2d(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return np.zeros(len(DESC_LIST_2D))

    desc_values = []
    desc_module = {name: func for name, func in Descriptors._descList}
    for desc_name in DESC_LIST_2D:
        try:
            func = desc_module[desc_name]
            val = func(mol)
            desc_values.append(val)
        except Exception:
            desc_values.append(0.0)
    return np.array(desc_values)


def preprocess_features(df_in: pd.DataFrame) -> np.ndarray:
    print("피처 생성 시작...")
    X_fp_morgan = np.stack(df_in['Drug'].apply(smiles_to_morgan_fingerprint).values)
    X_fp_maccs = np.stack(df_in['Drug'].apply(smiles_to_maccs_keys).values)
    X_desc = np.stack(df_in['Drug'].apply(smiles_to_rdkit_descriptors_2d).values)
    X_desc = np.nan_to_num(X_desc.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    X_combined = np.concatenate([X_fp_morgan, X_fp_maccs, X_desc], axis=1)
    print(f"피처 생성 완료. 최종 피처 형태: {X_combined.shape}")
    return X_combined


# 모델 학습
def train(train_data_path: str, save_path: str):
    print(f"학습 시작: {train_data_path}")

    # 1. 데이터 로드 및 전처리
    df_train = pd.read_csv(train_data_path)
    if 'Drug_ID' in df_train.columns:
        df_train = df_train.groupby('Drug_ID').agg({'Drug': 'first', 'Y': 'mean'}).reset_index()

    X_train = preprocess_features(df_train)
    y_train = df_train['Y'].values

    # 최적 파라미터 수정 필요 <----------------------------------
    best_params = {
        'n_estimators': 2000,
        'learning_rate': 0.025,
        'max_depth': 9,
        'subsample': 0.85,
        'colsample_bytree': 0.8,
        'gamma': 1e-08,
        'min_child_weight': 4
    }
    print(f"사용될 고정 하이퍼파라미터: {best_params}")

    # 3. 최종 모델 학습
    print("\n지정된 하이퍼파라미터로 최종 모델 학습 중...")
    final_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        tree_method='hist',
        **best_params
    )
    final_model.fit(X_train, y_train)

    # 4. 학습 데이터에 대한 성능 평가
    preds = final_model.predict(X_train)
    rmse = np.sqrt(mean_squared_error(y_train, preds))
    r2 = r2_score(y_train, preds)
    spearman_corr, _ = spearmanr(y_train, preds)

    print("\n--- 최종 모델 학습 성능 ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"R-squared: {r2:.4f}")
    print(f"Spearman Correlation: {spearman_corr:.4f}")

    # 5. 모델 저장
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    joblib.dump(final_model, save_path)
    print(f"\n모델 저장 완료: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model with fixed hyperparameters.")
    parser.add_argument('--train_data', type=str, default='calici_train.csv',
                        help='Path to the training data CSV file.')
    parser.add_argument('--save_path', type=str, default='model.pkl', help='Path to save the trained model.')
    args = parser.parse_args()
    train(train_data_path=args.train_data, save_path=args.save_path)