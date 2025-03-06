# employee-loan-flask-deployment
This is ML Loan Flask Development for apply employees loan is approved or not.

# Git Clone....
git clone https://github.com/ 

git config --global user.name "jroshanjha"
git config --global user.email "jroshan731@gmail.com"

## Create Virtual Environments & activate:- 
python -m venv python_env
venv/Scripts/activate

### Emplyees Loan Amount Prediction Database:- http://www.kagle.com/datasets/

# Install dependencies
pip install -r requirements.txt


## Targets variables:-  loan_status 

## Independent Variables:-
person_age	person_gender	person_education	person_income	person_emp_exp	person_home_ownership	loan_amnt	loan_intent

brew install git-lfs              # or download from https://git-lfs.github.com/
git lfs install
git lfs track "models/trained_model.pkl"  -- ".pkl"
git add .gitattributes
git add models/trained_model.pkl 
git commit -m "Add trained model with Git LFS"
git push origin development

git lfs push --all origin main


✅ Solution 1: Pull the Latest Changes Before Pushing
git pull origin main
git push origin main
✅ Solution 2: Force Push (If You Want to Overwrite Remote Changes)
⚠ Warning: This will overwrite any remote changes and may cause data loss for others.
Use only if you're sure that your local changes should replace the remote ones.
git push --force origin main


# Install Git LFS
git lfs install  

# Track large files (e.g., .h5, .pkl, .pt, .model)
git lfs track "*.h5" "*.pkl" "*.pt" "*.model"

# Add, commit, and push as usual
git add .gitattributes
git add model.h5
git commit -m "Add ML model"
git push origin main