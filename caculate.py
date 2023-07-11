import json
import random

gt_list=[]
pred_list=[]
temp_list=[]
use_gt_id=[]

with open('','r',encoding='utf-8') as f:
    with open('', 'r', encoding='utf-8') as fp:
            for d in f:
                mp=json.loads(d)
                temp_list.append(mp['label'])

            pred_split=fp.read().split('###')
            for idx, pred_ in enumerate(pred_split[1:]):
                if ('Yes' in pred_ or 'No' in pred_):
                    if 'Yes' in pred_:
                        pred_list.append('yes')
                    else:
                        pred_list.append('no')
                    gt_list.append(temp_list[idx])


from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score

print(gt_list)
print(pred_list)

print(accuracy_score(gt_list,pred_list))
print(precision_score(gt_list,pred_list,pos_label='yes'))
print(recall_score(gt_list,pred_list,pos_label='yes'))
print(f1_score(gt_list,pred_list,pos_label='yes'))

##############################################################################################
#40 epoch
#0.5691056910569106
#0.5641025641025641
#0.9705882352941176
#0.7135135135135134

#30 epoch
#0.5299145299145299
#0.5304347826086957
#0.9838709677419355
#0.6892655367231639

#25 epoch
#0.5471698113207547
#0.5454545454545454
#0.9473684210526315
#0.6923076923076923

#19 epoch
#0.5752212389380531
#0.5825242718446602
#0.9230769230769231
#0.7142857142857142

#5 epoch
#0.4351851851851852
#0.47619047619047616
#0.3389830508474576
#0.39603960396039606

#10 epoch
#0.4945054945054945
#0.5178571428571429
#0.6041666666666666
#0.5576923076923077

#epoch 15
#0.5321100917431193
#0.53125
#0.8947368421052632
#0.6666666666666666