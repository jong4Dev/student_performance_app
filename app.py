import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
from sklearn.inspection import permutation_importance

# Set page config
st.set_page_config(
    page_title="Student Performance Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
        .main {background-color: #f5f5f5;}
        .header {color: #2a3f5f;}
        .feature-card {
            border-radius: 5px; 
            padding: 15px; 
            background: white; 
            margin-bottom: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .prediction-card {
            border-radius: 10px;
            padding: 20px;
            background: white;
            margin: 10px 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .success {
            background-color: #e8f5e9;
            color: #2e7d32;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            font-size: 1.3rem;
            font-weight: bold;
        }
        .danger {
            background-color: #ffebee;
            color: #c62828;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            font-size: 1.3rem;
            font-weight: bold;
        }
        .metric-value {
            font-size: 1.5rem;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("📊 Student Performance Prediction Dashboard")
st.markdown("""
    This dashboard predicts student performance based on various factors.
    **Note:** Display shows 0=Pass, 1=Fail (model internally uses opposite convention)
""")

# Load data function
@st.cache_data
def load_data():
    df_math = pd.read_csv('data/student-mat.csv', sep=';')
    df_por = pd.read_csv('data/student-por.csv', sep=';')
    
    df_math['subject'] = 'Math'
    df_por['subject'] = 'Portuguese'
    df_combined = pd.concat([df_math, df_por], ignore_index=True)
    
    # Preprocessing (keeping original model logic)
    df_combined['lulus'] = df_combined['G3'].apply(lambda x: 1 if x >= 10 else 0)  # 1=Pass, 0=Fail
    
    categorical_cols = [
        'school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob',
        'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
        'nursery', 'higher', 'internet', 'romantic', 'subject'
    ]
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_combined[col] = le.fit_transform(df_combined[col])
    
    X = df_combined.drop(['G1', 'G2', 'G3', 'lulus'], axis=1)
    y = df_combined['lulus']
    
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    
    # Model training
    params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    grid = GridSearchCV(rf, params, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    
    best_rf = grid.best_estimator_
    
    return df_combined, X_test, y_test, best_rf, X.columns

# Load data
df_combined, X_test, y_test, best_rf, feature_columns = load_data()

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("About")
    st.markdown("""
        Student performance data from Portugal.
        - Math: 395 students
        - Portuguese: 649 students
    """)
    
    st.header("Key Metrics")
    accuracy = accuracy_score(y_test, best_rf.predict(X_test))
    st.metric("Model Accuracy", f"{accuracy:.1%}")
    
    st.header("Navigation")
    analysis_section = st.radio(
        "Go to section:",
        [
            "Data Overview",
            "Model Evaluation",
            "Feature Importance",
            "Performance Prediction"
        ]
    )

# ==================== MAIN CONTENT ====================

if analysis_section == "Data Overview":
    st.header("📊 Data Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Summary")
        st.markdown(f"""
            <div class="feature-card">
                <p><strong>Total Records:</strong> {len(df_combined):,}</p>
                <p><strong>Math Students:</strong> {len(df_combined[df_combined['subject']==0]):,}</p>
                <p><strong>Portuguese Students:</strong> {len(df_combined[df_combined['subject']==1]):,}</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Pass/Fail Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(x='lulus', data=df_combined, ax=ax, palette="Set2")
        ax.set_title('Distribution (Model: 1=fail, 0=pass)', pad=20)
        ax.set_xlabel("")
        ax.set_ylabel("Count")
        st.pyplot(fig)
    
    with col2:
        st.subheader("Subject Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(x='subject', data=df_combined, ax=ax, palette="pastel")
        ax.set_title('Subject Distribution (0=Math, 1=Portuguese)', pad=20)
        ax.set_xlabel("")
        ax.set_ylabel("Count")
        st.pyplot(fig)

elif analysis_section == "Model Evaluation":
    st.header("📈 Model Evaluation")
    
    st.subheader("Optimal Parameters")
    with st.expander("View best parameters"):
        st.code("""
        {
            'max_depth': None,
            'max_features': 'log2',
            'min_samples_leaf': 1,
            'min_samples_split': 2,
            'n_estimators': 300
        }
        """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Classification Report")
        y_pred = best_rf.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.2f}").background_gradient(cmap="Blues"))
    
    with col2:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Fail', 'Pass'],  # Model's perspective
            yticklabels=['Fail', 'Pass'],  # Model's perspective
            ax=ax
        )
        ax.set_title('Confusion Matrix (Model View)', pad=15)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    
    st.subheader("ROC Curve")
    y_prob = best_rf.predict_proba(X_test)[:, 1]  # Probability of class 1 (Pass)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='#4e79a7', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic', pad=15)
    ax.legend(loc="lower right")
    st.pyplot(fig)

elif analysis_section == "Feature Importance":
    st.header("🔍 Feature Importance")
    
    tab1, tab2 = st.tabs(["Gini Importance", "Permutation Importance"])
    
    with tab1:
        st.subheader("Top 20 Features (Gini Importance)")
        importance = best_rf.feature_importances_
        feature_importance = pd.DataFrame(
            {'Feature': X_test.columns, 'Importance': importance}
        ).sort_values('Importance', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(
            x='Importance',
            y='Feature',
            data=feature_importance.head(20),
            ax=ax,
            palette="viridis"
        )
        ax.set_title('Top 20 Features (Gini Importance)', pad=15)
        st.pyplot(fig)
    
    with tab2:
        st.subheader("Top 20 Features (Permutation Importance)")
        result = permutation_importance(
            best_rf, X_test, y_test, n_repeats=10, random_state=42
        )
        perm_importance = pd.DataFrame(
            {'Feature': X_test.columns, 'Importance': result.importances_mean}
        ).sort_values('Importance', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(
            x='Importance',
            y='Feature',
            data=perm_importance.head(20),
            ax=ax,
            palette="magma"
        )
        ax.set_title('Top 20 Features (Permutation Importance)', pad=15)
        st.pyplot(fig)

elif analysis_section == "Performance Prediction":
    st.header("🔮 Performance Prediction")
    st.markdown("""
        Predict whether a student will pass or fail based on their characteristics.
        **Display Note:** Results show 0=Pass, 1=Fail (opposite of model's internal convention)
    """)

    # Create mapping dictionaries
    CATEGORICAL_MAPPINGS = {
        'school': {'GP': 0, 'MS': 1},
        'sex': {'F': 0, 'M': 1},
        'address': {'U': 0, 'R': 1},
        'famsize': {'LE3': 0, 'GT3': 1},
        'Pstatus': {'T': 0, 'A': 1},
        'Mjob': {'teacher': 0, 'health': 1, 'services': 2, 'at_home': 3, 'other': 4},
        'Fjob': {'teacher': 0, 'health': 1, 'services': 2, 'at_home': 3, 'other': 4},
        'reason': {'home': 0, 'reputation': 1, 'course': 2, 'other': 3},
        'guardian': {'mother': 0, 'father': 1, 'other': 2},
        'schoolsup': {'yes': 0, 'no': 1},
        'famsup': {'yes': 0, 'no': 1},
        'paid': {'yes': 0, 'no': 1},
        'activities': {'yes': 0, 'no': 1},
        'nursery': {'yes': 0, 'no': 1},
        'higher': {'yes': 0, 'no': 1},
        'internet': {'yes': 0, 'no': 1},
        'romantic': {'yes': 0, 'no': 1},
        'subject': {'Math': 0, 'Portuguese': 1}
    }

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Personal Information")
            school = st.selectbox("School", list(CATEGORICAL_MAPPINGS['school'].keys()))
            sex = st.selectbox("Gender", list(CATEGORICAL_MAPPINGS['sex'].keys()))
            age = st.number_input("Age", 15, 22, 17)
            address = st.selectbox("Address", list(CATEGORICAL_MAPPINGS['address'].keys()))
            famsize = st.selectbox("Family Size", list(CATEGORICAL_MAPPINGS['famsize'].keys()))
            Pstatus = st.selectbox("Parents' Status", list(CATEGORICAL_MAPPINGS['Pstatus'].keys()))
            famrel = st.slider("Family Relationship (1-5)", 1, 5, 4)
            freetime = st.slider("Free Time (1-5)", 1, 5, 3)
        
        with col2:
            st.subheader("Family Background")
            Medu = st.selectbox("Mother's Education", [0, 1, 2, 3, 4], 
                              format_func=lambda x: ["None", "Primary", "Middle", "High", "Higher"][x])
            Fedu = st.selectbox("Father's Education", [0, 1, 2, 3, 4], 
                              format_func=lambda x: ["None", "Primary", "Middle", "High", "Higher"][x])
            Mjob = st.selectbox("Mother's Job", list(CATEGORICAL_MAPPINGS['Mjob'].keys()))
            Fjob = st.selectbox("Father's Job", list(CATEGORICAL_MAPPINGS['Fjob'].keys()))
            reason = st.selectbox("School Choice Reason", list(CATEGORICAL_MAPPINGS['reason'].keys()))
            guardian = st.selectbox("Guardian", list(CATEGORICAL_MAPPINGS['guardian'].keys()))
            goout = st.slider("Going Out (1-5)", 1, 5, 3)
            Dalc = st.slider("Workday Alcohol (1-5)", 1, 5, 1)
        
        with col3:
            st.subheader("Academic Factors")
            traveltime = st.selectbox("Travel Time", [1, 2, 3, 4], 
                                    format_func=lambda x: ["<15min", "15-30min", "30-60min", ">60min"][x-1])
            studytime = st.selectbox("Study Time", [1, 2, 3, 4], 
                                   format_func=lambda x: ["<2h", "2-5h", "5-10h", ">10h"][x-1])
            failures = st.slider("Past Failures", 0, 4, 0)
            schoolsup = st.selectbox("School Support", list(CATEGORICAL_MAPPINGS['schoolsup'].keys()))
            famsup = st.selectbox("Family Support", list(CATEGORICAL_MAPPINGS['famsup'].keys()))
            paid = st.selectbox("Paid Classes", list(CATEGORICAL_MAPPINGS['paid'].keys()))
            Walc = st.slider("Weekend Alcohol (1-5)", 1, 5, 1)
            health = st.slider("Health (1-5)", 1, 5, 3)
        
        st.subheader("Additional Information")
        col4, col5, col6 = st.columns(3)
        with col4:
            activities = st.selectbox("Activities", list(CATEGORICAL_MAPPINGS['activities'].keys()))
            nursery = st.selectbox("Attended Nursery", list(CATEGORICAL_MAPPINGS['nursery'].keys()))
        with col5:
            higher = st.selectbox("Wants Higher Ed", list(CATEGORICAL_MAPPINGS['higher'].keys()))
            internet = st.selectbox("Internet Access", list(CATEGORICAL_MAPPINGS['internet'].keys()))
        with col6:
            romantic = st.selectbox("In Relationship", list(CATEGORICAL_MAPPINGS['romantic'].keys()))
            subject = st.selectbox("Subject", list(CATEGORICAL_MAPPINGS['subject'].keys()))
            absences = st.number_input("Absences", 0, 93, 0)
        
        submitted = st.form_submit_button("Predict Performance")

    if submitted:
        # Convert all inputs to numeric
        input_numeric = {
            'school': CATEGORICAL_MAPPINGS['school'][school],
            'sex': CATEGORICAL_MAPPINGS['sex'][sex],
            'age': age,
            'address': CATEGORICAL_MAPPINGS['address'][address],
            'famsize': CATEGORICAL_MAPPINGS['famsize'][famsize],
            'Pstatus': CATEGORICAL_MAPPINGS['Pstatus'][Pstatus],
            'Medu': Medu,
            'Fedu': Fedu,
            'Mjob': CATEGORICAL_MAPPINGS['Mjob'][Mjob],
            'Fjob': CATEGORICAL_MAPPINGS['Fjob'][Fjob],
            'reason': CATEGORICAL_MAPPINGS['reason'][reason],
            'guardian': CATEGORICAL_MAPPINGS['guardian'][guardian],
            'traveltime': traveltime,
            'studytime': studytime,
            'failures': failures,
            'schoolsup': CATEGORICAL_MAPPINGS['schoolsup'][schoolsup],
            'famsup': CATEGORICAL_MAPPINGS['famsup'][famsup],
            'paid': CATEGORICAL_MAPPINGS['paid'][paid],
            'activities': CATEGORICAL_MAPPINGS['activities'][activities],
            'nursery': CATEGORICAL_MAPPINGS['nursery'][nursery],
            'higher': CATEGORICAL_MAPPINGS['higher'][higher],
            'internet': CATEGORICAL_MAPPINGS['internet'][internet],
            'romantic': CATEGORICAL_MAPPINGS['romantic'][romantic],
            'famrel': famrel,
            'freetime': freetime,
            'goout': goout,
            'Dalc': Dalc,
            'Walc': Walc,
            'health': health,
            'absences': absences,
            'subject': CATEGORICAL_MAPPINGS['subject'][subject]
        }

        # Create DataFrame
        input_df = pd.DataFrame([input_numeric])[feature_columns]

        # Make prediction
        prediction = best_rf.predict(input_df)
        proba = best_rf.predict_proba(input_df)[0]  # [P(Fail), P(Pass)]
        
        # Reverse display only (not changing model logic)
        display_pass = prediction[0] == 0  # Model's 1=Pass becomes our display Pass
        display_fail = prediction[0] == 1  # Model's 0=Fail becomes our display Fail
        
        # Probabilities (note: proba[0]=P(Fail), proba[1]=P(Pass) in model)
        pass_prob = proba[0] * 100
        fail_prob = proba[1] * 100

        # Display results
        st.markdown("---")
        st.subheader("Prediction Results")
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.markdown("""
                <div class="prediction-card">
                    <h3>Prediction</h3>
            """, unsafe_allow_html=True)
            
            if display_pass:
                st.markdown("""
                    <div class="success">✅ PASS (G3 ≥ 10)</div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="danger">❌ FAIL (G3 < 10)</div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_res2:
            st.markdown("""
                <div class="prediction-card">
                    <h3>Probability</h3>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <p>Probability Pass: <span class="metric-value">{pass_prob:.1f}%</span></p>
                <p>Probability Fail: <span class="metric-value">{fail_prob:.1f}%</span></p>
            """, unsafe_allow_html=True)
            
            # Visual comparison
            fig, ax = plt.subplots(figsize=(8, 2))
            bars = ax.barh(['Pass', 'Fail'], [pass_prob, fail_prob], color=['#4CAF50', '#F44336'])
            ax.set_xlim(0, 100)
            ax.set_title('Probability Comparison')
            
            # Add percentage labels
            for bar in bars:
                width = bar.get_width()
                ax.text(width/2, bar.get_y() + bar.get_height()/2,
                        f'{width:.1f}%',
                        ha='center', va='center', color='white', fontweight='bold')
            
            st.pyplot(fig)
            
            st.markdown("</div>", unsafe_allow_html=True)

        # Show important features
        st.subheader("Key Influencing Factors")
        
        importance_scores = best_rf.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': importance_scores,
            'Value': input_df.iloc[0].values,
            'Actual Value': [
                {v: k for k, v in CATEGORICAL_MAPPINGS.get(col, {}).items()}.get(val, val)
                for col, val in zip(feature_columns, input_df.iloc[0].values)
            ]
        }).sort_values('Importance', ascending=False).head(10)
        
        # Plot feature importance
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            x='Importance', 
            y='Feature', 
            data=feature_importance, 
            ax=ax, 
            palette="rocket"
        )
        ax.set_title('Top 10 Influential Features')
        st.pyplot(fig)
        
        # Show actual values
        st.write("Actual values for important features:")
        display_df = feature_importance[['Feature', 'Actual Value']].copy()
        st.dataframe(display_df.set_index('Feature'))

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: gray;">
        <p>Student Performance Prediction Dashboard • Created with Streamlit</p>
    </div>
""", unsafe_allow_html=True)