import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, precision_score, recall_score, roc_curve
from sklearn.tree import plot_tree
import pandas as pd
import phik
from phik import resources, report
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import shap

def divide_data(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y

def plot_phik(data, figsize=(12, 8)):
    phik_matrix = data.phik_matrix()
    plt.figure(figsize=(10, 8))
    sns.heatmap(phik_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.show()

def plot_hist_numeric(data, feature, figsize=(8, 4), x_min=None, x_max=None):
    filtered_data = data.copy()
    if x_min is not None:
        filtered_data = filtered_data[filtered_data[feature] >= x_min]
    if x_max is not None:
        filtered_data = filtered_data[filtered_data[feature] <= x_max]
    
    plt.figure(figsize=figsize)
    plt.grid()
    sns.histplot(filtered_data[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()

def plot_hist_categorical(data, feature, figsize=(4, 4)):
    category_counts = data[feature].value_counts()
    category_counts = category_counts.sort_values(ascending=False)
    plt.figure(figsize=figsize) 
    plt.grid()
    sns.barplot(x=category_counts.values, y=category_counts.index, palette="viridis", orient='h')
    plt.title(f'Distribution of {feature}')
    plt.ylabel(feature)
    plt.xlabel('Frequency')
    plt.show()

def plot_categorical_relationship(df, col1, col2):
    # Абсолютные значения
    count_crosstab = pd.crosstab(df[col1], df[col2])

    # Доли по строкам (внутри col1)
    row_prop = pd.crosstab(df[col1], df[col2], normalize='index')

    # Доли по столбцам (внутри col2)
    col_prop = pd.crosstab(df[col1], df[col2], normalize='columns')

    # Фигура с 3 подграфиками по горизонтали
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    # 1. Абсолютные значения
    sns.heatmap(count_crosstab, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_title(f'Абсолютные значения\n{col1} vs {col2}')
    axes[0].set_xlabel(col2)
    axes[0].set_ylabel(col1)

    # 2. Доли внутри col1 (по строкам)
    sns.heatmap(row_prop, annot=True, fmt=".2f", cmap="Greens", ax=axes[1])
    axes[1].set_title(f'Доли внутри {col1} (по строкам)')
    axes[1].set_xlabel(col2)
    axes[1].set_ylabel(col1)

    # 3. Доли внутри col2 (по столбцам)
    sns.heatmap(col_prop, annot=True, fmt=".2f", cmap="Oranges", ax=axes[2])
    axes[2].set_title(f'Доли внутри {col2} (по столбцам)')
    axes[2].set_xlabel(col2)
    axes[2].set_ylabel(col1)

    plt.tight_layout()
    plt.show()

def plot_numeric_relationship(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    target_col: str = None,
    target_colors: dict = None,
    x_min: float = None,
    x_max: float = None,
    y_min: float = None,
    y_max: float = None
):
    """
    Строит scatter plot зависимости между двумя числовыми переменными.
    При наличии бинарной таргетной переменной — точки окрашиваются по её значению.
    Позволяет задать ограничения на оси X и Y.

    :param df: pandas DataFrame
    :param x_col: Название числовой переменной по оси X
    :param y_col: Название числовой переменной по оси Y
    :param target_col: (опционально) Название бинарной переменной для окраски точек
    :param target_colors: (опционально) Словарь вида {значение_таргета: цвет}
    :param x_min: (опционально) Минимальное значение оси X
    :param x_max: (опционально) Максимальное значение оси X
    :param y_min: (опционально) Минимальное значение оси Y
    :param y_max: (опционально) Максимальное значение оси Y
    """
    # Проверка колонок
    for col in [x_col, y_col, target_col] if target_col else [x_col, y_col]:
        if col not in df.columns:
            raise ValueError(f"Колонка '{col}' отсутствует в DataFrame.")

    # Проверка типов
    if not pd.api.types.is_numeric_dtype(df[x_col]):
        raise TypeError(f"{x_col} не является числовой переменной.")
    if not pd.api.types.is_numeric_dtype(df[y_col]):
        raise TypeError(f"{y_col} не является числовой переменной.")

    # Проверка бинарного таргета
    if target_col is not None:
        unique_vals = sorted(df[target_col].dropna().unique())
        if len(unique_vals) != 2:
            raise ValueError(f"Таргет '{target_col}' должен быть бинарным (2 уникальных значения).")

        # Палитра
        if target_colors is None:
            palette = {unique_vals[0]: 'blue', unique_vals[1]: 'red'}
        else:
            if not all(val in target_colors for val in unique_vals):
                raise ValueError(f"target_colors должен содержать оба значения таргета: {unique_vals}")
            palette = target_colors

    # Построение графика
    plt.figure(figsize=(8, 6))
    if target_col:
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=target_col, palette=palette)
        plt.legend(title=target_col)
    else:
        sns.scatterplot(data=df, x=x_col, y=y_col, color='blue')

    # Ограничения осей
    if x_min is not None or x_max is not None:
        plt.xlim(left=x_min, right=x_max)
    if y_min is not None or y_max is not None:
        plt.ylim(bottom=y_min, top=y_max)

    plt.title(f'Зависимость {y_col} от {x_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def evaluate_classification(y_test, y_pred, y_probs=None, model_name="Model"):
    """
    Evaluate classification performance with comprehensive metrics and visualizations
    
    Parameters:
    -----------
    y_test : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_probs : array-like, optional
        Predicted probabilities for positive class (required for ROC AUC)
    model_name : str, optional
        Name of the model for display purposes
    """
    # Calculate metrics
    metrics = {
        'ROC AUC': roc_auc_score(y_test, y_probs) if y_probs is not None else None,
        'F1 Score': f1_score(y_test, y_pred, average='macro'), # используем macro усреднение, так как важны оба класса
        'Precision': precision_score(y_test, y_pred, average='macro'),
        'Recall': recall_score(y_test, y_pred, average='macro'),
        'Accuracy': (y_pred == y_test).mean()
    }
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Create figure
    plt.figure(figsize=(15, 6))
    
    # Plot 1: Confusion Matrix
    plt.subplot(1, 2, 1)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.title(f'{model_name} - Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    # Plot 2: ROC Curve (only if probabilities are provided)
    if y_probs is not None:
        fpr, tpr, thresholds = roc_curve(y_test, y_probs)
        plt.subplot(1, 2, 2)
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (AUC = {metrics["ROC AUC"]:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic', fontsize=14)
        plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.show()
    
    # Create metrics table
    metrics_df = pd.DataFrame({
        'Metric': list(metrics.keys()),
        'Value': [f'{val:.4f}' if isinstance(val, (int, float)) else 'N/A' 
                 for val in metrics.values()]
    })
    
    # Classification report
    class_report = {
        'Class': ['Positive', 'Negative'],
        'Precision': [
            precision_score(y_test, y_pred, pos_label=1),
            precision_score(y_test, y_pred, pos_label=0)
        ],
        'Recall': [
            recall_score(y_test, y_pred, pos_label=1),
            recall_score(y_test, y_pred, pos_label=0)
        ]
    }
    class_report_df = pd.DataFrame(class_report)
    
    # Display results
    print("\n" + "="*60)
    print(f"{model_name.upper()} EVALUATION".center(60))
    print("="*60)
    
    print("\nMAIN METRICS:")
    print(metrics_df.to_string(index=False))
    
    print("\n\nCLASSIFICATION REPORT:")
    print(class_report_df.to_string(index=False))
    
    print("\n" + "="*60)
    
    return metrics

def plot_feature_importance(model, feature_names, top_n=None, figsize=(10, 6), 
                           model_type='auto'):
    """
    Plot feature importance for various model types using Seaborn.
    
    Parameters:
    - model: Trained model (DecisionTree, RandomForest, LogisticRegression, etc.)
    - feature_names: List of feature names
    - top_n: Show only top N important features (None for all)
    - figsize: Figure size
    - model_type: 'auto' (default), 'tree', or 'linear'. If 'auto', tries to determine automatically
    """
    # Determine model type if auto
    if model_type == 'auto':
        if hasattr(model, 'feature_importances_'):
            model_type = 'tree'
        elif hasattr(model, 'coef_'):
            model_type = 'linear'
        else:
            raise ValueError("Could not determine model type automatically. Please specify 'tree' or 'linear'")
    
    # Get feature importances based on model type
    if model_type == 'tree':
        importances = model.feature_importances_
        importance_label = "Feature Importance"
    elif model_type == 'linear':
        # For linear models, use absolute coefficients as importance
        if len(model.coef_.shape) > 1:  # multi-class
            importances = np.mean(np.abs(model.coef_), axis=0)
        else:  # binary classification
            importances = np.abs(model.coef_[0])
        importance_label = "Absolute Coefficient"
    else:
        raise ValueError("model_type must be either 'tree' or 'linear'")
    
    # Create DataFrame
    feature_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Select top_n features if specified
    if top_n is not None:
        feature_imp = feature_imp.head(top_n)
    
    # Plot
    plt.figure(figsize=figsize)
    sns.barplot(x='Importance', y='Feature', data=feature_imp, palette='viridis')
    plt.title(f'Feature Importances ({model_type} model)')
    plt.xlabel(importance_label)
    plt.tight_layout()
    plt.show()
    
    return feature_imp

def visualize_decision_tree(model, feature_names, class_names=None, 
                           figsize=(20, 10), max_depth=None):
    """
    Visualize the decision tree structure.
    
    Parameters:
    - model: Trained DecisionTree model
    - feature_names: List of feature names
    - class_names: List of class names (for classification)
    - figsize: Figure size
    - max_depth: Maximum depth to display (None for full tree)
    """
    plt.figure(figsize=figsize)
    plot_tree(model, 
              feature_names=feature_names,
              class_names=class_names,
              filled=True,
              rounded=True,
              proportion=True,
              max_depth=max_depth)
    plt.title('Decision Tree Visualization')
    plt.show()