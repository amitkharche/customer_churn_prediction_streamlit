
# Churn Model Deployment Plan

## 1. Business Integration
- Integrate with CRM to flag high-risk customers weekly
- Feed predictions to retention team dashboard

## 2. Technical Stack
- API using Flask
- Streamlit app for visualization
- Scheduled batch inference pipeline

## 3. Monitoring Plan
- Track model drift
- Monthly recalibration
- Accuracy monitoring in production

## 4. Governance
- Explainability using SHAP
- Bias checks
- Secure access controls
