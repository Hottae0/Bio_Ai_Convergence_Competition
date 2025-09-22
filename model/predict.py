import pandas as pd
import numpy as np
import joblib
import argparse
import warnings
from typing import List

# RDKit 라이브러리
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, DataStructs, MACCSkeys

warnings.filterwarnings('ignore')

N_BITS_MORGAN = 2048


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


# --- 예측 ---
def predict(test_data_path: str, model_path: str) -> List[float]:
    print(f"예측 시작: {test_data_path}")
    model = joblib.load(model_path)
    df_test = pd.read_csv(test_data_path)

    print("테스트 데이터 전처리 중...")
    X_test = preprocess_features(df_test)

    y_pred = model.predict(X_test)
    predictions_list = y_pred.tolist()

    print("예측 완료.")
    return predictions_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict using a trained model.")
    parser.add_argument('--test_data', type=str, required=True, help='Path to the test data CSV file.')
    parser.add_argument('--model_path', type=str, default='model.pkl', help='Path to the trained model file.')
    parser.add_argument('--output_path', type=str, default='submission.csv',
                        help='Path to save the prediction results.')
    args = parser.parse_args()

    predictions = predict(test_data_path=args.test_data, model_path=args.model_path)

    submission_df = pd.DataFrame({'prediction': predictions})
    submission_df.to_csv(args.output_path, index=False)

    print(f"예측 결과 저장 완료: {args.output_path}")