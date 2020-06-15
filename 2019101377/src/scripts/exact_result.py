f=open('log_glove_nrms.txt','r')
#f=open('log_bert_nrms2.txt','r')
for line in f:
	if 'train info' in line or 'eval info' in line or 'group_auc' in line:
	#if  'eval info' in line or 'group_auc' in line:
		print(line)