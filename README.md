# Acute Malnutrition Early Warning System

This is a Streamlit app to forecast acute malnutrition admissions and classify seasonal risk levels.

## Features
- Upload CSV data
- Forecast future admissions
- Seasonal risk classification (Normal, Watch, Alert, Emergency)
- Visualizations: plots, correlations, risk maps
- Feature importance display
## How to Run Locally

1. **Clone the repository:**

```bash
git clone https://github.com/Awiosarah/malnutrition-ews.git
cd malnutrition-ews
Create a Python environment:

conda create -n streamlit-env python=3.11 -y
conda activate streamlit-env
Install dependencies:

pip install -r requirements.txt
Run the Streamlit app:

streamlit run app.py
Open in browser:

Streamlit will show a URL like http://localhost:8501