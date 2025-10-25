# PredictHR

## **Problem Statement**
Employees are the backbone of any organization, and retaining top talent is critical for operational success. High attrition rates lead to increased costs, loss of expertise, and reduced productivity. Factors driving employee attrition include:

- Inadequate compensation
- Limited career growth opportunities
- Poor supervision and management
- Desire to join globally recognized companies
- Lack of recognition and appreciation
- Restrictive organizational culture
- Underutilization of individual skills and talents

Predicting which employees are at risk of leaving can empower HR teams to take proactive measures, such as retention strategies or preventive hiring, to minimize organizational impact.

---

## **Objective**
The main goals of this project are to:

- Analyze factors affecting employee retention, including salary, satisfaction, growth opportunities, organizational policies, recognition, and employee feedback.
- Identify patterns and key contributors to attrition.
- Provide actionable insights to help organizations retain critical talent.

This study helps organizations understand **where they are falling short in retaining employees** and which areas require improvement.

---

## **Hypothesis**
1. Employee attrition increases the costs associated with recruitment, hiring, and training replacements.  
2. Employee attrition negatively impacts productivity and profitability within industries.

---

## **Approach**

**Decision Tree Modeling**  
- Built a Decision Tree classifier to predict employee attrition.  
- The dataset is highly imbalanced with 1223 'No' and 237 'Yes' entries.  
- Applied stratified sampling to maintain the proportion of attrition in training data.

**Model Evaluation**  
- Used 5-fold cross-validation to assess model generalization on unseen data.  
- The model was trained on stratified sample data and tested on unlabelled data.

**ROC Curve & Metrics**  
- ROC curve illustrates the trade-off between true positive rate and false positive rate.  
- Model Performance: ROC AUC = 0.6128, F1 Score = 0.81

---

## **Suggested Actions**

- **Enhance Work Conditions**  
  Offer flexible work arrangements, remote options, or ergonomic office spaces to improve employee satisfaction and work-life balance.

- **Competitive Compensation & Benefits**  
  Provide equitable salaries and perks like flexible schedules, travel benefits, and performance incentives to retain top talent.

- **Increase Employee Engagement**  
  Encourage skill development, provide growth opportunities, and involve employees in meaningful projects to increase job satisfaction and reduce attrition.

---

## **Technologies Used**
- **Programming Language:** Python  
- **Web Framework:** Streamlit  
- **Data Analysis:** Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn, Plotly  
- **Machine Learning:** scikit-learn (Decision Tree Classifier)  
- **Other:** Pillow, openpyxl  

---

## **Outcome**
The PredictHR dashboard allows HR professionals to:

- Explore employee data visually  
- Understand trends and correlations related to attrition  
- Predict whether employees are at risk of leaving  
- Take proactive measures to improve retention strategies
