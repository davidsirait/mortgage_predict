# testing pickle


import pickle


with open('models/randomforest.pkl', 'rb') as f:
    model = pickle.load(f)

final_feature = list(x_train.columns)
final_feature

# create data
cek = {"income": 1e10, 'loan': 1e8, 'gender': 'Female', 'education': 'SMA', 'cc': 'Yes', 'tenor': 100,
       'married': 'Yes'}
cek_df = pd.DataFrame(0, columns=final_feature, index=[0])

cek_df

cek_df['modIncomelog'] = np.log(cek['income'] + 1)
cek_df['modLoanlog'] = np.log(cek['loan'] + 1)
cek_df['Loan_Amount_Term'] = cek.get('tenor')
if cek.get('gender') == 'Male':
    cek_df['Gender_Male'] = 1
else:
    cek_df['Gender_Female'] = 1

if cek.get('education') in ['SMP', 'SMA', 'SMK', 'D3', 'D4', 'S1']:
    cek_df['Education_Not Graduate'] = 1
else:
    cek_df['Education_Graduate'] = 1

if cek.get('cc') == 'Yes':
    cek_df['Have_CC_Yes'] = 1
else:
    cek_df['Have_CC_No'] = 1

if cek.get('married') == 'Yes':
    cek_df['Married_Yes'] = 1
else:
    cek_df['Married_No'] = 1

cek_df

score = model.predict_proba(cek_df)[:, 1] * 100

int(score)