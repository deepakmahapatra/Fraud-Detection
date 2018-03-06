

import pandas as pd

class PreProcess:
	def __init__(self):
		pass
	def pre_processing(self, path):
		data = pd.read_csv(path)
		data = pd.get_dummies(data,columns=['TransactionCategoryId','TransactionPaymentTypeId'])
		data.drop(columns = ['TransactionDate','IsExpense','Description','Description','IncomeExpenseReportId','CaseFileReportId','AccountId'],inplace=True)
		data.fillna(value = 0,inplace=True)
		return data
