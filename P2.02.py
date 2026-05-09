import pandas as pd
import numpy as np
import math
import os
import multiprocessing as mp
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits


N_JOBS = max(1, (os.cpu_count() or 2) - 1)
RESULTS_DIR = "results"
GRID_SCORES_PATH = os.path.join(RESULTS_DIR, "P2.02_grid_scores.csv")


def save_grid_scores(grid, dataset_name, classifier_name, test_accuracy, output_path=GRID_SCORES_PATH):
    """Save one row per parameter combination from GridSearchCV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    results = pd.DataFrame(grid.cv_results_)
    param_cols = sorted([col for col in results.columns if col.startswith("param_")])
    split_cols = sorted([col for col in results.columns if col.startswith("split") and col.endswith("_test_score")])
    score_cols = split_cols + ["mean_test_score", "std_test_score", "rank_test_score"]

    output = results[param_cols + score_cols].copy()
    output.insert(0, "classifier", classifier_name)
    output.insert(0, "dataset", dataset_name)
    output["is_best"] = output["rank_test_score"] == 1
    output["final_test_accuracy"] = np.where(output["is_best"], test_accuracy, np.nan)
    output = output.sort_values(["dataset", "classifier", "rank_test_score"])

    write_header = not os.path.exists(output_path)
    output.to_csv(output_path, mode="a", header=write_header, index=False)
    print(f"Saved parameter-combination scores to {output_path}")


def make_grid_search(estimator, param_grid, cv):
    return GridSearchCV(
        estimator,
        param_grid,
        cv=cv,
        scoring='accuracy',
        verbose=2,
        n_jobs=N_JOBS,
        pre_dispatch=N_JOBS,
    )


def run_combined_experiments(X, y, dataset_name="Original", classifier_mode="all"):
    print(f"\n{'='*50}")
    print(f"--- Running [Filter + Lasso] on {dataset_name} Dataset ---")

    classifier_mode = classifier_mode.lower()
    valid_modes = {"logreg", "lda", "svm", "all"}
    if classifier_mode not in valid_modes:
        raise ValueError(f"classifier_mode must be one of {sorted(valid_modes)}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    # 1. data scaling (Standardization)
    # Could refactor to do this in pipeline if desired
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 2. 5 CV for Grid Search
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 3. pipeline
    pipeline = Pipeline([
        ('filter', SelectKBest(score_func=f_classif)),
        ('lasso', LogisticRegression(l1_ratio=1.0, solver='saga', max_iter=2000, random_state=42)),
    ])

    ## Update pipeline for LogReg (include scaler and add SelectFromModel / second LogReg)
    ## Want to compare running unconstrained logreg after using initial logreg for parameter selection
    log_reg_pipeline_2 = Pipeline([
        ('filter', SelectKBest(score_func=f_classif)),
        ('lasso', SelectFromModel(
            LogisticRegression(l1_ratio=1.0, solver='saga', max_iter=2000, random_state=42)
        )),
        #No lambda (1/lambda = c = C = np.inf) #So no L1 Ratio
        ('regression', LogisticRegression(solver='lbfgs', max_iter=2000, random_state=42, C=1e8))
    ])

    ## Pipeline for LDAs
    lda_pipeline = Pipeline([
        ('filter', SelectKBest(score_func=f_classif)),
        ('lasso', SelectFromModel(
            LogisticRegression(l1_ratio=1.0, solver='saga', max_iter=2000, random_state=42)
        )),
        ('lda', LinearDiscriminantAnalysis(solver="svd"))
    ])

    ## Pipeline for SVM
    svm_pipeline = Pipeline([
        ('filter', SelectKBest(score_func=f_classif)),
        ('lasso', SelectFromModel(
            LogisticRegression(l1_ratio=1.0, solver='saga', max_iter=10000, random_state=42)
        )),
        ('svm', SVC(random_state=42))
    ])

    # 4. Grid Search
    param_grid = {
        'filter__k':[100, 200, 500, 1000], 
        'lasso__C':[0.01, 0.1, 1, 10]
    }

    param_grid_log_reg_2 = {
        'filter__k': [50, 100, 200, 500, 1_000, 2_000, 3_000],

        'lasso__estimator__C': [
            #0.001,
            #0.01,
            #0.1,
            #1,
            10
        ]
    }

    ##param_grid for LDA
    ## Can possibly refactor this later to run two pipelines (One for feature select, then each other one for specific model)
    ## May save time, but may get differesults if different numbers of features better for each model

    param_grid_lda = {
        'filter__k': [50, 100, 200, 500, 1_000, 2_000, 3_000],

        'lasso__estimator__C': [
            #0.001,
            #0.01,
            #0.1,
            #1,
            10
        ]
    }

    ##param_grid for SVM
    param_grid_svm = {
        'filter__k': [600, 1500],

        'lasso__estimator__C': [
            #0.001,
            #0.01,
            # 0.1,
            # 1,
            10
        ],

        'svm__C': [1, 10],
        'svm__kernel': ['rbf']
    }
    
    selected_pixels_by_classifier = {}

    if classifier_mode in ("logreg", "all"):
        print("Running GridSearchCV to find optimal combination of Filter 'k' and Lasso 'C' on Logistic Regression...")
        grid = make_grid_search(log_reg_pipeline_2, param_grid_log_reg_2, cv)
        grid.fit(X_train_scaled, y_train)
        
        # 5. best model evaluation
        best_model = grid.best_estimator_
        print(f"Optimal Filter 'k': {grid.best_params_['filter__k']}")
        print(f"Optimal Lasso 'C': {grid.best_params_['lasso__estimator__C']}")
        print(f"CV Accuracy (on 75% train data): {grid.best_score_:.4f}")
        
        test_accuracy = best_model.score(X_test_scaled, y_test)
        print(f"FINAL TEST ACCURACY (on 25% unseen data): {test_accuracy:.4f}")
        save_grid_scores(grid, dataset_name, "logreg", test_accuracy)

        filter_mask = best_model.named_steps['filter'].get_support()
        filter_selected_pixels_in_original = np.where(filter_mask)[0] 
        
        selector = best_model.named_steps['lasso']
        lasso_mask = selector.get_support()
        
        final_selected_pixels = filter_selected_pixels_in_original[lasso_mask]
        selected_pixels_by_classifier["logreg"] = final_selected_pixels
        print(f"Final number of features selected: {len(final_selected_pixels)}")
        print(f"{'='*50}")

    #6.
    #Grid Search for LDA


    if classifier_mode in ("lda", "all"):
        print("Running GridSearchCV to find optimal combination of Filter 'k' and Lasso 'C' on LDA...")
        grid = make_grid_search(lda_pipeline, param_grid_lda, cv)
        grid.fit(X_train_scaled, y_train)
        
        # 7. best model evaluation for LDA
        best_model = grid.best_estimator_
        print(f"Optimal Filter 'k': {grid.best_params_['filter__k']}")
        print(f"Optimal Lasso 'C': {grid.best_params_['lasso__estimator__C']}")
        print(f"CV Accuracy (on 75% train data): {grid.best_score_:.4f}")
        
        test_accuracy = best_model.score(X_test_scaled, y_test)
        print(f"FINAL TEST ACCURACY (on 25% unseen data): {test_accuracy:.4f}")
        save_grid_scores(grid, dataset_name, "lda", test_accuracy)

        filter_mask = best_model.named_steps['filter'].get_support()
        filter_selected_pixels_in_original = np.where(filter_mask)[0] 
        
        selector = best_model.named_steps['lasso']
        lasso_mask = selector.get_support()
        
        final_selected_pixels = filter_selected_pixels_in_original[lasso_mask]
        selected_pixels_by_classifier["lda"] = final_selected_pixels
        
        print(f"Final number of features selected: {len(final_selected_pixels)}")
        print(f"{'='*50}")

    #8.
    #Grid Search for SVM

    if classifier_mode in ("svm", "all"):
        print("Running GridSearchCV to find optimal combination of Filter 'k', Lasso 'C', and SVM params on SVM...")
        grid = make_grid_search(svm_pipeline, param_grid_svm, cv)
        grid.fit(X_train_scaled, y_train)
        
        # 9. best model evaluation for SVM
        best_model = grid.best_estimator_
        print(f"Optimal Filter 'k': {grid.best_params_['filter__k']}")
        print(f"Optimal Lasso 'C': {grid.best_params_['lasso__estimator__C']}")
        print(f"Optimal SVM 'C': {grid.best_params_['svm__C']}")
        print(f"Optimal SVM kernel: {grid.best_params_['svm__kernel']}")
        print(f"CV Accuracy (on 75% train data): {grid.best_score_:.4f}")
        
        test_accuracy = best_model.score(X_test_scaled, y_test)
        print(f"FINAL TEST ACCURACY (on 25% unseen data): {test_accuracy:.4f}")
        save_grid_scores(grid, dataset_name, "svm", test_accuracy)

        filter_mask = best_model.named_steps['filter'].get_support()
        filter_selected_pixels_in_original = np.where(filter_mask)[0] 
        
        selector = best_model.named_steps['lasso']
        lasso_mask = selector.get_support()
        
        final_selected_pixels = filter_selected_pixels_in_original[lasso_mask]
        selected_pixels_by_classifier["svm"] = final_selected_pixels
        
        print(f"Final number of features selected: {len(final_selected_pixels)}")
        print(f"{'='*50}")
    
    return selected_pixels_by_classifier

# ==========================================
# main
# ==========================================
if __name__ == "__main__":
    mp.freeze_support()
    CLASSIFIER_MODE = "svm"  #################### Choose from: "logreg", "lda", "svm", "all" ########################################

    print("Loading data...")
    PATHIM = "data//cnd_large//images.csv" 
    PATHLB = "data//cnd_large//labels.csv"

    X_df = pd.read_csv(PATHIM, sep=",", index_col=0)
    y_df = pd.read_csv(PATHLB, sep=",", index_col=0)

    y_df = y_df.rename(columns={"0":"label"})

    X = X_df.values
    y = y_df.values.ravel() 
    n_samples, n_features = X.shape

    # Theme 2
    print("\nProcessing Theme 2: Flipping half of the images upside down...")
    side_len = int(math.sqrt(n_features))
    img_height, img_width = side_len, side_len

    X_flipped = X.copy()
    # Note: reshape() does NOT create a copy, they share the same memory.
    X_flipped_3d = X_flipped.reshape(n_samples, img_height, img_width)
    np.random.seed(42) 
    
    for cls in np.unique(y):
        cls_indices = np.where(y == cls)[0]
        np.random.shuffle(cls_indices)
        indices_to_flip = cls_indices[:len(cls_indices) // 2]
        # Since memory is shared, modifying X_flipped_3d automatically updates the 2D X_flipped.
        # No need to reshape it back!
        X_flipped_3d[indices_to_flip] = X_flipped_3d[indices_to_flip, ::-1, :]

    print(f"Using {N_JOBS} parallel worker(s) for GridSearchCV.")
    if os.path.exists(GRID_SCORES_PATH):
        os.remove(GRID_SCORES_PATH)

    with threadpool_limits(limits=1):
        # Part 1
        final_pixels_original = run_combined_experiments(
            X, y, dataset_name="Original", classifier_mode=CLASSIFIER_MODE
        )

        # Part 2
        final_pixels_flipped = run_combined_experiments(
            X_flipped, y, dataset_name="Half-Flipped", classifier_mode=CLASSIFIER_MODE
        )
    
    # comparison of selected pixels
    print("\n" + "="*50)
    print("Comparison: Original vs Flipped")
    print("="*50)
    for classifier_name in final_pixels_original:
        print(f"\nClassifier: {classifier_name}")
        print(f"Pixels needed (Original): {len(final_pixels_original[classifier_name])}")
        print(f"Pixels needed (Flipped):  {len(final_pixels_flipped[classifier_name])}")
        common_pixels = np.intersect1d(
            final_pixels_original[classifier_name],
            final_pixels_flipped[classifier_name]
        )
        print(f"Overlapping selected pixels: {len(common_pixels)}")
