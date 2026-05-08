import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def run_combined_experiments(X, y, dataset_name="Original"):
    print(f"\n{'='*50}")
    print(f"--- Running [Filter + Lasso] on {dataset_name} Dataset ---")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    # 1. data scaling (Standardization)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 2. 5 CV for Grid Search
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 3. pipeline
    pipeline = Pipeline([
        ('filter', SelectKBest(score_func=f_classif)),
        ('lasso', LogisticRegression(l1_ratio=1.0, solver='liblinear', max_iter=2000, random_state=42))
    ])
    
    # 4. Grid Search
    param_grid = {
        'filter__k':[100, 200, 500, 1000], 
        'lasso__C':[0.01, 0.1, 1, 10]
    }
    
    print("Running GridSearchCV to find optimal combination of Filter 'k' and Lasso 'C'...")
    grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train_scaled, y_train)
    
    # 5. best model evaluation
    best_model = grid.best_estimator_
    print(f"Optimal Filter 'k': {grid.best_params_['filter__k']}")
    print(f"Optimal Lasso 'C': {grid.best_params_['lasso__C']}")
    print(f"CV Accuracy (on 75% train data): {grid.best_score_:.4f}")
    
    test_accuracy = best_model.score(X_test_scaled, y_test)
    print(f"FINAL TEST ACCURACY (on 25% unseen data): {test_accuracy:.4f}")

    filter_mask = best_model.named_steps['filter'].get_support()
    filter_selected_pixels_in_original = np.where(filter_mask)[0] 
    
    lasso_coef = best_model.named_steps['lasso'].coef_[0]
    lasso_nonzero_indices = np.where(lasso_coef != 0)[0] 
    
    final_selected_pixels = filter_selected_pixels_in_original[lasso_nonzero_indices]
    print(f"Final number of features selected: {len(final_selected_pixels)}")
    print(f"{'='*50}")
    
    return final_selected_pixels

# ==========================================
# main
# ==========================================
if __name__ == "__main__":
    print("Loading data...")
    X_df = pd.read_csv('cnd_large\images.csv', index_col=0, header=None) 
    y_df = pd.read_csv('cnd_large\labels.csv', index_col=0, header=None)

    X = X_df.values
    y = y_df.values.ravel() 
    n_samples, n_features = X.shape

    # Part 1
    final_pixels_original = run_combined_experiments(X, y, dataset_name="Original")

    # Theme 2
    print("\nProcessing Theme 2: Flipping half of the images upside down...")
    side_len = int(math.sqrt(n_features))
    img_height, img_width = side_len, side_len

    X_flipped = X.copy()
    np.random.seed(42) 
    
    for cls in np.unique(y):
        cls_indices = np.where(y == cls)[0]
        np.random.shuffle(cls_indices)
        indices_to_flip = cls_indices[:len(cls_indices) // 2]
        for idx in indices_to_flip:
            img_2d = X_flipped[idx].reshape(img_height, img_width)
            X_flipped[idx] = np.flipud(img_2d).flatten()

    final_pixels_flipped = run_combined_experiments(X_flipped, y, dataset_name="Half-Flipped")
    
    # comparison of selected pixels
    print("\n" + "="*50)
    print("Comparison: Original vs Flipped")
    print("="*50)
    print(f"Pixels needed (Original): {len(final_pixels_original)}")
    print(f"Pixels needed (Flipped):  {len(final_pixels_flipped)}")
    common_pixels = np.intersect1d(final_pixels_original, final_pixels_flipped)
    print(f"Overlapping selected pixels: {len(common_pixels)}")
