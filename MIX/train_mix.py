
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, confusion_matrix
import optuna
from optuna.samplers import TPESampler
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置日志
def setup_logger():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"pred_mix_seg_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

logger = setup_logger()

# 参数解析
def parse_args():
    parser = argparse.ArgumentParser(description='Lung US Classification Pipeline')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, 
                       default='/root/data/lungUS_cls/mix_pred_seg_csv/',
                       help='Directory containing the fold data')
    
    # 特征选择参数
    parser.add_argument('--feature_selection', type=str, 
                       default='lasso', choices=['lasso', 'elasticnet', 'none'],
                       help='Feature selection method')
    parser.add_argument('--correlation_threshold', type=float, 
                       default=0.9, help='Threshold for removing correlated features')
    
    # 模型参数
    parser.add_argument('--model', type=str, 
                       default='xgboost', choices=['xgboost', 'svm', 'randomforest'],
                       help='Model to use for classification')
    
    # 训练参数
    parser.add_argument('--inner_folds', type=int, 
                       default=10, help='Number of inner CV folds')
    parser.add_argument('--n_trials', type=int, 
                       default=50, help='Number of Optuna trials')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, 
                       default='./pred_mask_mix_results/seg',
                       help='Directory to save prediction results')
    
    return parser.parse_args()

# 计算额外指标
def calculate_additional_metrics(y_true, y_pred_prob):
    """计算ACC、灵敏度、特异性"""
    y_pred = (y_pred_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)  # 灵敏度
    specificity = tn / (tn + fp)  # 特异性
    
    return acc, sensitivity, specificity

# 数据加载和预处理
class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.irrelevant_cols = [
            # "patient_id", "image",
            "augmentation",
            "diagnostics_Image-original_Hash",
            "diagnostics_Image-original_Minimum",
            "diagnostics_Image-original_Maximum",
            "diagnostics_Mask-original_Hash",
            "diagnostics_Mask-original_BoundingBox",
            "diagnostics_Mask-original_CenterOfMassIndex",
            "diagnostics_Mask-original_CenterOfMass",
            "diagnostics_Versions_PyRadiomics",
            "diagnostics_Versions_Numpy",
            "diagnostics_Versions_SimpleITK",
            "diagnostics_Versions_PyWavelet",
            "diagnostics_Versions_Python",
            "diagnostics_Configuration_Settings",
            "diagnostics_Configuration_EnabledImageTypes",
            "diagnostics_Image-original_Dimensionality",
            "diagnostics_Image-original_Spacing",
            "diagnostics_Image-original_Size",
            "diagnostics_Mask-original_Size",
            "diagnostics_Mask-original_BoundingBox",
            "diagnostics_Mask-original_Spacing",
            "diagnostics_Mask-original_CenterOfMassIndex",
            "diagnostics_Mask-original_CenterOfMass"
        ]
    
    def load_fold_data(self, fold):
        train_path = os.path.join(self.data_dir, f'fold{fold}_train_labeled.csv')
        test_path = os.path.join(self.data_dir, f'fold{fold}_test_labeled.csv')
        
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        logger.info(f"\n{'='*50}")
        logger.info(f"📊 Fold {fold} Data Summary:")
        logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        logger.info(f"Train class distribution:\n{train_df['label'].value_counts()}")
        logger.info(f"Test class distribution:\n{test_df['label'].value_counts()}")
        
        return train_df, test_df
    
    def preprocess_data(self, df):
        # 保存标识列
        id_cols = ['patient_id', 'image']
        existing_id_cols = [col for col in id_cols if col in df.columns]
        ids = df[existing_id_cols].copy()
        
        # 移除不相关列
        cols_to_drop = [col for col in self.irrelevant_cols if col in df.columns]
        df = df.drop(columns=cols_to_drop, errors='ignore')
        
        # 分离特征和标签
        label_col = 'label'
        X = df.drop(columns=[label_col] + existing_id_cols)
        y = df[label_col]
        
        # 仅保留数值类型特征
        numeric_cols = X.select_dtypes(include=['number']).columns
        X = X[numeric_cols]
        
        logger.info(f"📊 预处理后维度: {X.shape}")
        logger.debug(f"剩余特征样例: {list(X.columns[:5])}...")  # 只打印前5个特征避免日志过长
        
        return ids, X, y

# 特征工程
class FeatureEngineer:
    def __init__(self, correlation_threshold=0.9):
        self.correlation_threshold = correlation_threshold
        self.scaler = StandardScaler()
    
    def remove_correlated_features(self, X):
        corr_matrix = X.corr().abs()
        
        # 生成布尔型上三角矩阵（修复点）
        upper_mask = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
        
        # 筛选高相关特征
        upper = corr_matrix.where(upper_mask)
        to_drop = [col for col in corr_matrix.columns 
                if any(upper[col] > self.correlation_threshold)]
        
        return X.drop(columns=to_drop)
    
    def select_features(self, X, y, method='lasso'):
        if method == 'none':
            return X
        
        if method == 'lasso':
            selector = Lasso(alpha=0.01, random_state=42)
        elif method == 'elasticnet':
            selector = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42)
        
        sfm = SelectFromModel(selector, threshold='1.25*median')
        X_selected = sfm.fit_transform(X, y)
        selected_features = X.columns[sfm.get_support()]
        
        logger.info(f"After {method} feature selection: {len(selected_features)} features selected")
        return pd.DataFrame(X_selected, columns=selected_features)
    
    def process_features(self, X_train, X_test, y_train, feature_selection_method):
        # 标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 转换为DataFrame
        X_train_processed = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_processed = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        # 移除高度相关特征
        X_train_reduced = self.remove_correlated_features(X_train_processed)
        common_cols = X_train_reduced.columns.intersection(X_test_processed.columns)
        X_test_reduced = X_test_processed[common_cols]
        
        # 特征选择
        X_train_selected = self.select_features(X_train_reduced, y_train, feature_selection_method)
        common_cols = X_train_selected.columns.intersection(X_test_reduced.columns)
        X_test_selected = X_test_reduced[common_cols]
        
        return X_train_selected, X_test_selected

# 模型训练和调优
class ModelTrainer:
    def __init__(self, model_type, inner_folds, n_trials):
        self.model_type = model_type
        self.inner_folds = inner_folds
        self.n_trials = n_trials
        self.best_params = None

        self.fixed_params = {
            # 'xgboost': {'random_state': 42,'scale_pos_weight': 0.588},
            'xgboost': {'random_state': 42},
            'randomforest': {'random_state': 42, 'bootstrap': True},
            'svm': {'probability': True, 'random_state': 42}
        }
    
    def objective(self, trial, X, y, groups):
        if self.model_type == 'xgboost':

            optuna_params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            
            }
            params = {**self.fixed_params['xgboost'], **optuna_params}
            model = XGBClassifier(**params)

        elif self.model_type == 'randomforest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'bootstrap': True,
                'random_state': 42
            }
            model = RandomForestClassifier(**params)
        elif self.model_type == 'svm':
            params = {
                'C': trial.suggest_float('C', 0.1, 10, log=True),
                'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf']),
                'gamma': trial.suggest_float('gamma', 1e-4, 1, log=True) if params['kernel'] == 'rbf' else 'auto',
                'probability': True,
                'random_state': 42
            }
            model = SVC(**params)
        
        # 内层交叉验证
        group_kfold = GroupKFold(n_splits=self.inner_folds)
        auc_scores = []
        
        for train_idx, val_idx in group_kfold.split(X, y, groups):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred)
            auc_scores.append(auc)
        
        return np.mean(auc_scores)
    
    def tune_model(self, X, y, groups):
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(lambda trial: self.objective(trial, X, y, groups), 
                        n_trials=self.n_trials)
        
        self.best_params = {
            **study.best_params,  
            **self.fixed_params.get(self.model_type, {}) 
        }
        
        logger.info(f"🎯 Best parameters: {self.best_params}")
        logger.info(f"Best AUC: {study.best_value:.4f}")
        
        return study.best_params
    
    def train_final_model(self, X_train, y_train, X_test, y_test):
        if self.model_type == 'xgboost':
            model = XGBClassifier(**self.best_params)
        elif self.model_type == 'randomforest':
            model = RandomForestClassifier(**self.best_params)
        elif self.model_type == 'svm':
            model = SVC(**self.best_params, probability=True)
        
        model.fit(X_train, y_train)
        
        # 训练集性能
        train_pred = model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, train_pred)
        
        # 测试集性能
        test_pred = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, test_pred)
        
        # 计算额外指标
        test_acc, test_sens, test_spec = calculate_additional_metrics(y_test, test_pred)
        
        logger.info(f"📊 Final Model Performance:")
        logger.info(f"Train AUC: {train_auc:.4f}")
        logger.info(f"Test AUC: {test_auc:.4f}, ACC: {test_acc:.4f}, Sensitivity: {test_sens:.4f}, Specificity: {test_spec:.4f}")
        
        return model, test_pred, test_auc, test_acc, test_sens, test_spec

# 主流程
def main():
    args = parse_args()
    logger.info(f"🚀 Starting pipeline with parameters:\n{args.__dict__}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    data_loader = DataLoader(args.data_dir)
    feature_engineer = FeatureEngineer(args.correlation_threshold)
    model_trainer = ModelTrainer(args.model, args.inner_folds, args.n_trials)
    
    fold_results = []
    all_test_results = []
    
    # 外层5折交叉验证
    for fold in range(1, 6):
        logger.info(f"\n{'='*50}")
        logger.info(f"🌟 Processing Fold {fold}")
        
        # 加载数据
        train_df, test_df = data_loader.load_fold_data(fold)
        
        # 预处理
        ids_train, X_train, y_train = data_loader.preprocess_data(train_df)
        ids_test, X_test, y_test = data_loader.preprocess_data(test_df)
        
        # 特征工程
        X_train_processed, X_test_processed = feature_engineer.process_features(
            X_train, X_test, y_train, args.feature_selection
        )
        
        # 确保同一个patient_id不会同时出现在训练和验证集
        groups = train_df['patient_id'].values
        
        # 模型调优
        logger.info("🔍 Starting hyperparameter tuning...")
        model_trainer.tune_model(X_train_processed, y_train, groups)
        
        # 训练最终模型
        logger.info("🏋️ Training final model...")
        model, test_pred, test_auc, test_acc, test_sens, test_spec = model_trainer.train_final_model(
            X_train_processed, y_train, 
            X_test_processed, y_test
        )
        
        # 保存测试集预测结果
        test_results = pd.DataFrame({
            'fold': fold,
            'patient_id': ids_test['patient_id'],
            'image': ids_test['image'],
            'true_label': y_test.values,
            'pred_prob': test_pred
        })
        csv_path = os.path.join(args.output_dir, f'fold{fold}_test_predictions.csv')
        test_results.to_csv(csv_path, index=False)
        logger.info(f"💾 Saved test predictions to: {csv_path}")
        
        # 记录当前折的结果
        fold_results.append({
            'fold': fold,
            'test_auc': test_auc,
            'test_acc': test_acc,
            'test_sens': test_sens,
            'test_spec': test_spec
        })
        
        # 保存所有结果用于最终汇总
        all_test_results.append(test_results)
    
    # 计算5折指标的平均值和标准差
    fold_metrics = pd.DataFrame(fold_results)
    metrics_summary = fold_metrics.describe().loc[['mean', 'std']]

    logger.info(f"\n{'='*50}")
    logger.info("📊 5-Fold Cross-Validation Metrics Summary:")
    logger.info(metrics_summary)

    # 保存详细指标和汇总
    detailed_metrics_path = os.path.join(args.output_dir, 'detailed_fold_metrics.csv')
    fold_metrics.to_csv(detailed_metrics_path, index=False)
    logger.info(f"💾 Saved detailed fold metrics to: {detailed_metrics_path}")

    summary_metrics_path = os.path.join(args.output_dir, 'summary_metrics.csv')
    metrics_summary.to_csv(summary_metrics_path)
    logger.info(f"💾 Saved summary metrics to: {summary_metrics_path}")

    # 最终结果报告
    logger.info(f"\n{'='*50}")
    logger.info("🎉 Final 5-Fold Performance:")
    logger.info(f"AUC: {metrics_summary.loc['mean', 'test_auc']:.4f} ± {metrics_summary.loc['std', 'test_auc']:.4f}")
    logger.info(f"ACC: {metrics_summary.loc['mean', 'test_acc']:.4f} ± {metrics_summary.loc['std', 'test_acc']:.4f}")
    logger.info(f"Sensitivity: {metrics_summary.loc['mean', 'test_sens']:.4f} ± {metrics_summary.loc['std', 'test_sens']:.4f}")
    logger.info(f"Specificity: {metrics_summary.loc['mean', 'test_spec']:.4f} ± {metrics_summary.loc['std', 'test_spec']:.4f}")

if __name__ == "__main__":
    main()
