import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import statsmodels.api as sm


# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------

st.set_page_config(
    page_title="Meadowlight Farm - Player Progression & Churn",
    layout="wide"
)

st.title("Meadowlight Farm: Player Progression & Churn")
st.caption(
    "Early warmup behavior analysis for player progression, churn, and reminder targeting."
)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

@st.cache_data
def load_data():
    warmup = pd.read_csv("meadowlight_warmup.csv")
    outcomes = pd.read_csv("meadowlight_outcomes.csv")

    warmup["start_date"] = pd.to_datetime(warmup["start_date"])

    return warmup, outcomes


warmup, outcomes = load_data()


# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------

@st.cache_data
def build_features(warmup):

    # 1. Struggle levels:
    # number of warmup levels requiring 5 or more tries
    struggle_levels = (
        warmup.assign(struggle=(warmup["total_tries"] >= 5).astype(int))
        .groupby("user_id")["struggle"]
        .sum()
        .reset_index(name="struggle_levels")
    )

    # 2. Total coin spending
    total_coin_spending = (
        warmup.groupby("user_id")["coin_spending"]
        .sum()
        .reset_index(name="total_coin_spending")
    )

    # 3. Completion days
    completion = (
        warmup.groupby("user_id")["start_date"]
        .agg(["min", "max"])
        .reset_index()
    )

    completion["completion_days"] = (
        completion["max"] - completion["min"]
    ).dt.days

    completion_days = completion[
        ["user_id", "completion_days"]
    ]

    # 4. Active days
    active_days = (
        warmup.groupby("user_id")["start_date"]
        .nunique()
        .reset_index(name="active_days")
    )

    # 5. Average tries on hard levels (difficulty 4 or 5)
    hard_levels = warmup[warmup["difficulty"] >= 4]

    hard_level_avg_tries = (
        hard_levels.groupby("user_id")["total_tries"]
        .mean()
        .reset_index(name="hard_level_avg_tries")
    )

    # Merge all engineered variables
    features = struggle_levels.merge(
        total_coin_spending, on="user_id"
    )

    features = features.merge(
        completion_days, on="user_id"
    )

    features = features.merge(
        active_days, on="user_id"
    )

    features = features.merge(
        hard_level_avg_tries, on="user_id"
    )

    return features


features = build_features(warmup)

model_data = features.merge(
    outcomes,
    on="user_id",
    how="inner"
)


feature_cols = [
    "struggle_levels",
    "total_coin_spending",
    "completion_days",
    "active_days",
    "hard_level_avg_tries"
]


# ---------------------------------------------------------
# TRAIN / TEST SPLIT
# ---------------------------------------------------------

train_data, test_data = train_test_split(
    model_data,
    test_size=0.20,
    random_state=12345
)

X_train = train_data[feature_cols]
X_test = test_data[feature_cols]

X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test, has_constant="add")


# ---------------------------------------------------------
# MODELS
# ---------------------------------------------------------

outcome_names = [
    "played_400",
    "played_500",
    "churn_15_28",
    "churn_22_28"
]


@st.cache_resource
def fit_models(train_data):

    logistic_models = {}
    tree_models = {}

    X_train = train_data[feature_cols]
    X_train_const = sm.add_constant(X_train)

    for outcome in outcome_names:

        y_train = train_data[outcome]

        logistic_models[outcome] = sm.Logit(
            y_train,
            X_train_const
        ).fit(disp=False)

        tree_models[outcome] = DecisionTreeClassifier(
            max_depth=3,
            random_state=12345
        )

        tree_models[outcome].fit(
            X_train,
            y_train
        )

    return logistic_models, tree_models


logistic_models, tree_models = fit_models(train_data)


# ---------------------------------------------------------
# HELPER FUNCTION
# ---------------------------------------------------------

def calculate_metrics(y_true, probabilities, threshold):

    pred = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        pred
    ).ravel()

    accuracy = accuracy_score(y_true, pred)

    sensitivity = recall_score(
        y_true,
        pred,
        zero_division=0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    precision = precision_score(
        y_true,
        pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        pred,
        zero_division=0
    )

    return {
        "Accuracy": accuracy,
        "Sensitivity / Recall": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "F1": f1,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }


# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    "1. Warmup Feature Engineering",
    "2. Prediction Models & Evaluation",
    "3. Managerial Insights"
])


# =========================================================
# TAB 1
# =========================================================

with tab1:

    st.header("Warmup Feature Engineering")

    st.write(
        "The original warmup data contains one row for each player-level "
        "combination across levels 100–199. These records were summarized "
        "into one row per player."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Warmup Records", f"{len(warmup):,}")
    c2.metric("Players", f"{features['user_id'].nunique():,}")
    c3.metric("Final Modeling Rows", f"{len(model_data):,}")

    st.subheader("Engineered Variables")

    feature_definition = pd.DataFrame({
        "Feature": [
            "struggle_levels",
            "total_coin_spending",
            "completion_days",
            "active_days",
            "hard_level_avg_tries"
        ],
        "Definition": [
            "Number of warmup levels requiring 5 or more tries.",
            "Total coins spent across warmup levels 100–199.",
            "Calendar days between the player's first and last warmup level date.",
            "Number of unique days on which the player was active during warmup.",
            "Average number of tries on difficulty 4–5 levels."
        ],
        "Behavioral Logic": [
            "Captures repeated difficulty or persistence challenges.",
            "Captures resource use and possible difficulty-related spending.",
            "Measures overall pace through the warmup.",
            "Measures consistency of engagement.",
            "Captures performance specifically on harder warmup content."
        ]
    })

    st.dataframe(
        feature_definition,
        use_container_width=True
    )

    st.subheader("Summary Statistics")

    st.dataframe(
        features[feature_cols].describe().round(2),
        use_container_width=True
    )

    st.subheader("Feature Distribution")

    selected_feature = st.selectbox(
        "Choose a feature",
        feature_cols
    )

    fig, ax = plt.subplots()

    ax.hist(
        features[selected_feature],
        bins=30
    )

    ax.set_title(
        f"Distribution of {selected_feature}"
    )

    ax.set_xlabel(selected_feature)
    ax.set_ylabel("Number of Players")

    st.pyplot(fig)

    st.subheader("Correlation Among Engineered Features")

    correlation = (
        features[feature_cols]
        .corr()
        .round(2)
    )

    st.dataframe(
        correlation,
        use_container_width=True
    )


# =========================================================
# TAB 2
# =========================================================

with tab2:

    st.header("Prediction Models and Evaluation")

    st.write(
        "The same 80/20 train-test split and random seed (12345) are used "
        "for all outcomes and both model families."
    )

    outcome = st.selectbox(
        "Select an outcome",
        outcome_names
    )

    threshold = st.slider(
        "Probability threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05
    )

    y_test = test_data[outcome]

    # Logistic probabilities
    logistic_prob = logistic_models[outcome].predict(
        X_test_const
    )

    # Tree probabilities
    tree_prob = tree_models[outcome].predict_proba(
        X_test
    )[:, 1]

    logistic_metrics = calculate_metrics(
        y_test,
        logistic_prob,
        threshold
    )

    tree_metrics = calculate_metrics(
        y_test,
        tree_prob,
        threshold
    )

    st.subheader("Model Comparison")

    comparison = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Sensitivity / Recall",
            "Specificity",
            "Precision",
            "F1",
            "ROC-AUC"
        ],
        "Logistic Regression": [
            logistic_metrics["Accuracy"],
            logistic_metrics["Sensitivity / Recall"],
            logistic_metrics["Specificity"],
            logistic_metrics["Precision"],
            logistic_metrics["F1"],
            roc_auc_score(
                y_test,
                logistic_prob
            )
        ],
        "Decision Tree": [
            tree_metrics["Accuracy"],
            tree_metrics["Sensitivity / Recall"],
            tree_metrics["Specificity"],
            tree_metrics["Precision"],
            tree_metrics["F1"],
            roc_auc_score(
                y_test,
                tree_prob
            )
        ]
    })

    st.dataframe(
        comparison.round(3),
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Logistic Regression Confusion Matrix")

        cm_logit = pd.DataFrame(
            [
                [
                    logistic_metrics["TN"],
                    logistic_metrics["FP"]
                ],
                [
                    logistic_metrics["FN"],
                    logistic_metrics["TP"]
                ]
            ],
            index=[
                "Actual 0",
                "Actual 1"
            ],
            columns=[
                "Predicted 0",
                "Predicted 1"
            ]
        )

        st.dataframe(
            cm_logit,
            use_container_width=True
        )

    with col2:

        st.subheader("Decision Tree Confusion Matrix")

        cm_tree = pd.DataFrame(
            [
                [
                    tree_metrics["TN"],
                    tree_metrics["FP"]
                ],
                [
                    tree_metrics["FN"],
                    tree_metrics["TP"]
                ]
            ],
            index=[
                "Actual 0",
                "Actual 1"
            ],
            columns=[
                "Predicted 0",
                "Predicted 1"
            ]
        )

        st.dataframe(
            cm_tree,
            use_container_width=True
        )

    st.subheader("Logistic Regression Results")

    logit_model = logistic_models[outcome]

    logistic_table = pd.DataFrame({
        "Coefficient": logit_model.params,
        "Odds Ratio": np.exp(
            logit_model.params
        ),
        "P-value": logit_model.pvalues
    })

    logistic_table["Significant (p < .05)"] = (
        logistic_table["P-value"] < 0.05
    )

    st.dataframe(
        logistic_table.round(4),
        use_container_width=True
    )

    st.subheader("Classification Tree")

    fig, ax = plt.subplots(
        figsize=(18, 8)
    )

    plot_tree(
        tree_models[outcome],
        feature_names=feature_cols,
        class_names=["No", "Yes"],
        filled=True,
        rounded=True,
        ax=ax
    )

    st.pyplot(fig)

    st.subheader("Tree Feature Importance")

    importance = pd.DataFrame({
        "Feature": feature_cols,
        "Importance":
            tree_models[outcome].feature_importances_
    }).sort_values(
        "Importance",
        ascending=False
    )

    st.dataframe(
        importance.round(3),
        use_container_width=True
    )


# =========================================================
# TAB 3
# =========================================================

with tab3:

    st.header("Managerial Insights")

    st.subheader("Main Findings")

    st.markdown(
        """
        - **Completion time is the clearest and most consistent signal.**
          Players who take longer to complete the warmup are less likely
          to progress to levels 400 and 500 and are more likely to churn.

        - **Active days are especially important for churn.**
          More active warmup days are associated with lower subsequent churn.

        - **Coin spending provides some progression signal.**
          Higher warmup coin spending was associated with lower odds
          of later progression in the logistic models.

        - Warmup variables predict **progression more effectively than churn**.
          The progression models achieved ROC-AUC values around 0.70–0.73,
          while churn models were much closer to 0.50.
        """
    )

    st.subheader("Recommended Models and Thresholds")

    recommendations = pd.DataFrame({
        "Outcome": [
            "played_400",
            "played_500",
            "churn_15_28",
            "churn_22_28"
        ],
        "Recommended Model": [
            "Logistic Regression",
            "Decision Tree",
            "Logistic Regression",
            "Logistic Regression"
        ],
        "Threshold": [
            0.50,
            0.40,
            0.30,
            0.40
        ],
        "Reason": [
            "High recall and strongest F1 balance.",
            "Best overall accuracy and F1 balance.",
            "Better churn recall than the tree at the selected threshold.",
            "Better overall balance while maintaining useful recall."
        ]
    })

    st.dataframe(
        recommendations,
        use_container_width=True
    )

    st.subheader("Reminder Strategy")

    st.info(
        """
        Recall should receive the greatest emphasis for churn-focused
        reminder targeting. A false negative means failing to contact
        a player who later disengages, potentially losing an opportunity
        to retain that player.

        A false positive means contacting a player who would have remained
        engaged anyway. This creates an unnecessary message and possible
        annoyance, but may be less costly than missing an at-risk player.
        """
    )

    st.warning(
        """
        The churn models should be used as an early screening tool rather
        than a perfect churn prediction system. Lower thresholds increase
        recall but also increase false positives.
        """
    )
