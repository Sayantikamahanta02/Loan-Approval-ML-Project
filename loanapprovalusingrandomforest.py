import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv('loan_approval.csv')
print(df)
lb=LabelEncoder()
df['Employment_Status']=lb.fit_transform(df['Employment_Status'])
df['Marital_Status']=lb.fit_transform(df['Marital_Status'])
df['Loan_Status']=lb.fit_transform(df['Loan_Status'])
#print(df)
x=df.drop(['Applicant_ID','Loan_Status'],axis=1)
y=df[['Loan_Status']]
print(x.shape)
print(y.shape)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=45)
model=RandomForestClassifier(n_estimators=4)
model.fit(x_train,y_train)
print(x_test)
print(x_test.shape)
print(y_test)
print(y_test.shape)
predict_result=model.predict(x_test)
print(predict_result)
acc_sore=accuracy_score(y_test,predict_result)
print("Accuracy score using knn:",acc_sore)
Age=int(input("Enter the age:"))
Income=int(input("Enter the Income:"))
Employment_Status=int(input("Enter the Employment_Status :"))
Credit_Score=int(input("Enter the Credit_Score :"))
Loan_Amount=int(input("Enter the Loan_Amount :"))
Loan_Term_Months=int(input("Enter the Loan_Term_Months :"))
Existing_Debt=int(input("Enter the Existing_Debt:"))
Marital_Status=int(input("Enter the Marital_Status:"))
user=[[Age,Income,Employment_Status,Credit_Score,Loan_Amount,Loan_Term_Months,Existing_Debt,Marital_Status]]
result=model.predict(user)
if result==0:
    print("Approve")
else:
    print("Rejected")    
