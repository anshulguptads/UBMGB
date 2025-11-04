import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.model_utils import train_and_evaluate, predict_new

st.set_page_config(page_title="Universal Bank: Personal Loan Prediction", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("data/UniversalBank.csv")
    df.columns = [c.strip() for c in df.columns]
    return df

def marketing_insights(df):
    st.subheader("Top 5 Marketing Insights (Charts)")
    # 1) Acceptance by Income decile
    df1 = df.copy()
    df1["IncomeBin"] = pd.qcut(df1["Income"], 10, duplicates="drop")
    agg1 = df1.groupby("IncomeBin")["Personal Loan"].mean().reset_index()
    fig1, ax1 = plt.subplots()
    ax1.plot(range(len(agg1)), agg1["Personal Loan"])
    ax1.set_xticks(range(len(agg1)))
    ax1.set_xticklabels([str(i) for i in agg1["IncomeBin"]], rotation=45, ha="right")
    ax1.set_xlabel("Income Decile")
    ax1.set_ylabel("Loan Acceptance Rate")
    ax1.set_title("Acceptance Rate by Income Decile")
    st.pyplot(fig1)

    # 2) Acceptance by Education
    if "Education" in df.columns:
        edu_map = {1:"Undergrad", 2:"Graduate", 3:"Advanced/Professional"}
        df2 = df.copy()
        df2["EducationLabel"] = df2["Education"].map(edu_map)
        agg2 = df2.groupby("EducationLabel")["Personal Loan"].mean().reset_index()
        fig2, ax2 = plt.subplots()
        ax2.bar(agg2["EducationLabel"], agg2["Personal Loan"])
        ax2.set_xlabel("Education Level")
        ax2.set_ylabel("Loan Acceptance Rate")
        ax2.set_title("Acceptance Rate by Education")
        st.pyplot(fig2)

    # 3) Acceptance by Family size
    if "Family" in df.columns:
        agg3 = df.groupby("Family")["Personal Loan"].mean().reset_index()
        fig3, ax3 = plt.subplots()
        ax3.bar(agg3["Family"], agg3["Personal Loan"])
        ax3.set_xlabel("Family Size")
        ax3.set_ylabel("Loan Acceptance Rate")
        ax3.set_title("Acceptance Rate by Family Size")
        st.pyplot(fig3)

    # 4) Online usage vs acceptance
    if "Online" in df.columns:
        agg4 = df.groupby("Online")["Personal Loan"].mean().reset_index()
        fig4, ax4 = plt.subplots()
        ax4.bar(agg4["Online"], agg4["Personal Loan"])
        ax4.set_xlabel("Online Banking (0/1)")
        ax4.set_ylabel("Loan Acceptance Rate")
        ax4.set_title("Acceptance Rate by Online Banking Usage")
        st.pyplot(fig4)

    # 5) Credit card spend (CCAvg) deciles vs acceptance
    df5 = df.copy()
    df5["CCBin"] = pd.qcut(df5["CCAvg"], 10, duplicates="drop")
    agg5 = df5.groupby("CCBin")["Personal Loan"].mean().reset_index()
    fig5, ax5 = plt.subplots()
    ax5.plot(range(len(agg5)), agg5["Personal Loan"])
    ax5.set_xticks(range(len(agg5)))
    ax5.set_xticklabels([str(i) for i in agg5["CCBin"]], rotation=45, ha="right")
    ax5.set_xlabel("Credit Card Spend Decile")
    ax5.set_ylabel("Loan Acceptance Rate")
    ax5.set_title("Acceptance Rate by Credit Card Spend")
    st.pyplot(fig5)

def show_training_evaluation(df):
    st.subheader("Train & Evaluate Models")
    with st.sidebar:
        st.markdown("### Settings")
        drop_zip = st.checkbox("Drop 'Zip code' from features", value=False)
        which = st.multiselect(
            "Select models to train",
            options=["Decision Tree","Random Forest","Gradient Boosted Trees"],
            default=["Decision Tree","Random Forest","Gradient Boosted Trees"]
        )
        run = st.button("Run Training/Evaluation")

    if run:
        fitted, results_df, roc_data, _ = train_and_evaluate(df, which, drop_zip=drop_zip)
        st.session_state["fitted_models"] = fitted
        st.session_state["results_df"] = results_df
        st.session_state["roc_data"] = roc_data
        st.session_state["drop_zip"] = drop_zip

    if "results_df" in st.session_state:
        st.markdown("**Metrics (Test set)**")
        st.dataframe(
            st.session_state["results_df"].sort_values("Testing Accuracy", ascending=False),
            use_container_width=True
        )

        # Confusion matrices as plain tables (no color)
        cm_keys = [k for k in st.session_state["fitted_models"].keys() if k.endswith("Confusion Matrix")]
        cols = st.columns(len(cm_keys)) if cm_keys else [st.container()]
        for i, key in enumerate(cm_keys):
            cm = st.session_state["fitted_models"][key]
            with cols[i]:
                st.markdown(f"**{key.replace(' Confusion Matrix','')} — Confusion Matrix (Test)**")
                st.table(pd.DataFrame(cm, index=["True: 0","True: 1"], columns=["Predicted: 0","Predicted: 1"]))

        # Combined ROC
        if st.session_state["roc_data"]:
            fig, ax = plt.subplots()
            for name, (fpr, tpr, roc_auc) in st.session_state["roc_data"].items():
                ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
            ax.plot([0,1], [0,1], linestyle="--")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curves (Test Set)")
            ax.legend(loc="lower right")
            st.pyplot(fig)

def show_predict_section(df):
    st.subheader("Predict on New Customer")
    st.caption("Enter details and click **Predict**. The predicted row can be appended and downloaded.")
    cols = st.columns(3)
    def num_input(col, label, minv=None, maxv=None, step=1, value=0):
        with col:
            return st.number_input(label, min_value=minv, max_value=maxv, step=step, value=value)
    age = num_input(cols[0], "Age", 0, 120, 1, 35)
    exp = num_input(cols[1], "Experience", 0, 60, 1, 10)
    inc = num_input(cols[2], "Income ($000)", 0, 1000, 1, 60)
    zipcode = num_input(cols[0], "Zip code", 0, 99999, 1, 94105)
    fam = num_input(cols[1], "Family", 0, 10, 1, 3)
    ccavg = num_input(cols[2], "CCAvg ($000)", 0, 1000, 1, 2)
    edu = num_input(cols[0], "Education (1=Undergrad,2=Graduate,3=Advanced)", 1, 3, 1, 2)
    mort = num_input(cols[1], "Mortgage ($000)", 0, 2000, 1, 0)
    sec = num_input(cols[2], "Securities (0/1)", 0, 1, 1, 0)
    cd = num_input(cols[0], "CDAccount (0/1)", 0, 1, 1, 0)
    online = num_input(cols[1], "Online (0/1)", 0, 1, 1, 1)
    cc = num_input(cols[2], "CreditCard (0/1)", 0, 1, 1, 1)

    features = {
        "Age": age, "Experience": exp, "Income": inc, "Zip code": zipcode, "Family": fam, "CCAvg": ccavg,
        "Education": edu, "Mortgage": mort, "Securities": sec, "CDAccount": cd, "Online": online, "CreditCard": cc
    }

    if st.button("Predict"):
        if "fitted_models" not in st.session_state:
            st.warning("Please train models first in the 'Train & Evaluate' tab.")
        else:
            first_model_name = [k for k in st.session_state["fitted_models"].keys() if not k.endswith("Confusion Matrix")][0]
            pipe = st.session_state["fitted_models"][first_model_name]
            pred, proba = predict_new(pipe, features, df, drop_zip=st.session_state.get("drop_zip", False))
            st.success(f"Prediction using **{first_model_name}**: {pred} (1=Interested, 0=Not Interested)")
            if proba is not None:
                st.info(f"Predicted probability of interest: {proba:.3f}")

            # Append the new predicted row to a copy and allow download
            new_row = features.copy()
            new_row["Personal Loan"] = pred
            out_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            st.download_button(
                label="Download dataset with NEW predicted row (CSV)",
                data=out_df.to_csv(index=False).encode("utf-8"),
                file_name="UniversalBank_with_prediction.csv",
                mime="text/csv"
            )

def main():
    st.title("Universal Bank — Personal Loan Prediction")
    st.caption("Head of Marketing dashboard: insights, model evaluation, and live predictions.")
    df = load_data()
    tabs = st.tabs(["Insights", "Train & Evaluate", "Predict"])
    with tabs[0]:
        marketing_insights(df)
    with tabs[1]:
        show_training_evaluation(df)
    with tabs[2]:
        show_predict_section(df)

if __name__ == "__main__":
    main()
