import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.feature_selection  import mutual_info_regression
from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import ensemble
feature_index = ['id','f1','f2','f3','f4','f5','f6','f7',',f8','f9','f10','f13','f14','f15','f16','f17','f18','f19','f20','f21','f22','f23','f24','f25','f26','f27','f28','f29','f30','f31','f32','f33','f34','f35','f36','f37','f38','f39','f40','f41','f42','f43','f44','f45','f46','f47','f48','f49','f50','f51','f52','f53','f54','f55','f56','f57','f58','f59','f60','f61','f62','f63','f64','f65','f66','f67','f68','f69','f70','f71','f72','f73','f74','f75','f76','f77','f78','f79','f80','f81','f82','f83','f84','f85','f86','f87','f88','f89','f90','f91','f92','f93','f94','f95','f96','f97','f98','f99','f100','f101','f102','f103','f104','f105','f106','f107','f108','f109','f110','f111','f112','f113','f114','f115','f116','f117','f118','f119','f120','f121','f122','f123','f124','f125','f126','f127','f128','f129','f130','f131','f132','f133','f134','f135','f136','f137','f138','f139','f140','f141','f142','f143','f144','f145','f146','f147','f148','f149','f150','f151','f152','f153','f154','f155','f156','f157','f158','f159','f160','f161','f162','f163','f164','f165','f166','f167','f168','f169','f170','f171','f172','f173','f174','f175','f176','f177','f178','f179','f180','f181','f182','f183','f184','f185','f186','f187','f188','f189','f190','f191','f192','f193','f194','f195','f196','f197','f198','f199','f200','f201','f202','f203','f204','f205','f206','f207','f208','f209','f210','f211','f212','f213','f214','f215','f216','f217','f218','f219','f220','f221','f222','f223','f224','f225','f226','f227','f228','f229','f230','f231','f232','f233','f234','f235','f236','f237','f238','f239','f240','f241','f242','f243','f244','f245','f246','f247','f248','f249','f250','f251','f252','f253','f254','f255','f256','f257','f258','f259','f260','f261','f262','f263','f264','f265','f266','f267','f268','f269','f270','f271','f272','f273','f274','f275','f276','f277','f278','f279','f280','f281','f282','f283','f284','f285','f286','f287','f288','f289','f290','f291','f292','f293','f294','f295','f296','f297','f298','f299','f300','f301','f302','f303','f304','f305','f306','f307','f308','f309','f310','f311','f312','f313','f314','f315','f316','f317','f318','f319','f320','f321','f322','f323','f324','f325','f326','f327','f328','f329','f330','f331','f332','f333','f334','f335','f336','f337','f338','f339','f340','f341','f342','f343','f344','f345','f346','f347','f348','f349','f350','f351','f352','f353','f354','f355','f356','f357','f358','f359','f360','f361','f362','f363','f364','f365','f366','f367','f368','f369','f370','f371','f372','f373','f374','f375','f376','f377','f378','f379','f380','f381','f382','f383','f384','f385','f386','f387','f388','f389','f390','f391','f392','f393','f394','f395','f396','f397','f398','f399','f400','f401','f402','f403','f404','f405','f406','f407','f408','f409','f410','f411','f412','f413','f414','f415','f416','f417','f418','f419','f420','f421','f422','f423','f424','f425','f426','f427','f428','f429','f430','f431','f432','f433','f434','f435','f436','f437','f438','f439','f440','f441','f442','f443','f444','f445','f446','f447','f448','f449','f450','f451','f452','f453','f454','f455','f456','f457','f458','f459','f460','f461','f464','f465','f466','f467','f468','f469','f470','f471','f472','f475','f476','f477','f478','f479','f480','f481','f482','f483','f484','f485','f486','f487','f488','f489','f490','f491','f492','f493','f494','f495','f496','f497','f498','f499','f500','f501','f502','f503','f504','f505','f506','f507','f508','f509','f510','f511','f512','f513','f514','f515','f516','f517','f518','f519','f520','f521','f522','f523','f524','f525','f526','f527','f528','f529','f530','f531','f532','f533','f534','f535','f536','f537','f538','f539','f540','f541','f542','f543','f544','f545','f546','f547','f548','f549','f550','f551','f552','f553','f554','f555','f556','f557','f558','f559','f560','f561','f562','f563','f564','f565','f566','f567','f568','f569','f570','f571','f572','f573','f574','f575','f576','f577','f578','f579','f580','f581','f582','f583','f584','f585','f586','f587','f588','f589','f590','f591','f592','f593','f594','f595','f596','f597','f598','f599','f600','f601','f604','f606','f607','f608','f609','f610','f611','f612','f613','f614','f615','f616','f617','f618','f619','f620','f621','f622','f623','f624','f625','f626','f627','f628','f629','f630','f631','f632','f633','f634','f635','f636','f637','f638','f639','f640','f641','f642','f643','f644','f645','f646','f647','f648','f649','f650','f651','f652','f653','f654','f655','f656','f657','f658','f659','f660','f661','f662','f663','f664','f665','f666','f667','f668','f669','f670','f671','f672','f673','f674','f675','f676','f677','f678','f679','f680','f681','f682','f683','f684','f685','f686','f687','f688','f689','f690','f691','f692','f693','f694','f695','f696','f697','f698','f699','f700','f701','f702','f703','f704','f705','f706','f707','f708','f709','f710','f711','f712','f713','f714','f715','f716','f717','f718','f719','f720','f721','f722','f723','f724','f725','f726','f727','f728','f729','f730','f731','f732','f733','f734','f735','f736','f737','f738','f739','f740','f741','f742','f743','f744','f745','f746','f747','f748','f749','f750','f751','f752','f753','f754','f755','f756','f757','f758','f759','f760','f761','f762','f763','f764','f765','f766','f767','f768','f769','f770','f771','f772','f773','f774','f775','f776','f777','f778']
feature_index2 = ['id','f1','f2','f3','f4','f5','f6','f7',',f8','f9','f10','f13','f14','f15','f16','f17','f18','f19','f20','f21','f22','f23','f24','f25','f26','f27','f28','f29','f30','f31','f32','f33','f34','f35','f36','f37','f38','f39','f40','f41','f42','f43','f44','f45','f46','f47','f48','f49','f50','f51','f52','f53','f54','f55','f56','f57','f58','f59','f60','f61','f62','f63','f64','f65','f66','f67','f68','f69','f70','f71','f72','f73','f74','f75','f76','f77','f78','f79','f80','f81','f82','f83','f84','f85','f86','f87','f88','f89','f90','f91','f92','f93','f94','f95','f96','f97','f98','f99','f100','f101','f102','f103','f104','f105','f106','f107','f108','f109','f110','f111','f112','f113','f114','f115','f116','f117','f118','f119','f120','f121','f122','f123','f124','f125','f126','f127','f128','f129','f130','f131','f132','f133','f134','f135','f136','f137','f138','f139','f140','f141','f142','f143','f144','f145','f146','f147','f148','f149','f150','f151','f152','f153','f154','f155','f156','f157','f158','f159','f160','f161','f162','f163','f164','f165','f166','f167','f168','f169','f170','f171','f172','f173','f174','f175','f176','f177','f178','f179','f180','f181','f182','f183','f184','f185','f186','f187','f188','f189','f190','f191','f192','f193','f194','f195','f196','f197','f198','f199','f200','f201','f202','f203','f204','f205','f206','f207','f208','f209','f210','f211','f212','f213','f214','f215','f216','f217','f218','f219','f220','f221','f222','f223','f224','f225','f226','f227','f228','f229','f230','f231','f232','f233','f234','f235','f236','f237','f238','f239','f240','f241','f242','f243','f244','f245','f246','f247','f248','f249','f250','f251','f252','f253','f254','f255','f256','f257','f258','f259','f260','f261','f262','f263','f264','f265','f266','f267','f268','f269','f270','f271','f272','f273','f274','f275','f276','f277','f278','f279','f280','f281','f282','f283','f284','f285','f286','f287','f288','f289','f290','f291','f292','f293','f294','f295','f296','f297','f298','f299','f300','f301','f302','f303','f304','f305','f306','f307','f308','f309','f310','f311','f312','f313','f314','f315','f316','f317','f318','f319','f320','f321','f322','f323','f324','f325','f326','f327','f328','f329','f330','f331','f332','f333','f334','f335','f336','f337','f338','f339','f340','f341','f342','f343','f344','f345','f346','f347','f348','f349','f350','f351','f352','f353','f354','f355','f356','f357','f358','f359','f360','f361','f362','f363','f364','f365','f366','f367','f368','f369','f370','f371','f372','f373','f374','f375','f376','f377','f378','f379','f380','f381','f382','f383','f384','f385','f386','f387','f388','f389','f390','f391','f392','f393','f394','f395','f396','f397','f398','f399','f400','f401','f402','f403','f404','f405','f406','f407','f408','f409','f410','f411','f412','f413','f414','f415','f416','f417','f418','f419','f420','f421','f422','f423','f424','f425','f426','f427','f428','f429','f430','f431','f432','f433','f434','f435','f436','f437','f438','f439','f440','f441','f442','f443','f444','f445','f446','f447','f448','f449','f450','f451','f452','f453','f454','f455','f456','f457','f458','f459','f460','f461','f464','f465','f466','f467','f468','f469','f470','f471','f472','f475','f476','f477','f478','f479','f480','f481','f482','f483','f484','f485','f486','f487','f488','f489','f490','f491','f492','f493','f494','f495','f496','f497','f498','f499','f500','f501','f502','f503','f504','f505','f506','f507','f508','f509','f510','f511','f512','f513','f514','f515','f516','f517','f518','f519','f520','f521','f522','f523','f524','f525','f526','f527','f528','f529','f530','f531','f532','f533','f534','f535','f536','f537','f538','f539','f540','f541','f542','f543','f544','f545','f546','f547','f548','f549','f550','f551','f552','f553','f554','f555','f556','f557','f558','f559','f560','f561','f562','f563','f564','f565','f566','f567','f568','f569','f570','f571','f572','f573','f574','f575','f576','f577','f578','f579','f580','f581','f582','f583','f584','f585','f586','f587','f588','f589','f590','f591','f592','f593','f594','f595','f596','f597','f598','f599','f600','f601','f604','f606','f607','f608','f609','f610','f611','f612','f613','f614','f615','f616','f617','f618','f619','f620','f621','f622','f623','f624','f625','f626','f627','f628','f629','f630','f631','f632','f633','f634','f635','f636','f637','f638','f639','f640','f641','f642','f643','f644','f645','f646','f647','f648','f649','f650','f651','f652','f653','f654','f655','f656','f657','f658','f659','f660','f661','f662','f663','f664','f665','f666','f667','f668','f669','f670','f671','f672','f673','f674','f675','f676','f677','f678','f679','f680','f681','f682','f683','f684','f685','f686','f687','f688','f689','f690','f691','f692','f693','f694','f695','f696','f697','f698','f699','f700','f701','f702','f703','f704','f705','f706','f707','f708','f709','f710','f711','f712','f713','f714','f715','f716','f717','f718','f719','f720','f721','f722','f723','f724','f725','f726','f727','f728','f729','f730','f731','f732','f733','f734','f735','f736','f737','f738','f739','f740','f741','f742','f743','f744','f745','f746','f747','f748','f749','f750','f751','f752','f753','f754','f755','f756','f757','f758','f759','f760','f761','f762','f763','f764','f765','f766','f767','f768','f769','f770','f771','f772','f773','f774','f775','f776','f777','f778','loss']
feature_pair_sub_list = [['f527','f528'],['f528','f274']]
#feature_pair_sub_list = [[520, 521], [271, 521], [271, 520], [67, 466], [623, 664], [7, 536], [66, 529], [561, 562], [248, 602], [570, 571], [218, 766], [64, 765], [208, 590], [423, 660], [312, 463], [290, 592], [621, 755], [52, 311], [65, 422], [350, 656], [278, 420], [320, 633], [507, 761], [0, 341], [139, 665], [10, 724], [53, 319], [367, 698], [279, 421], [9, 358], [48, 287], [375, 653], [397, 728], [197, 666], [38, 295], [402, 758], [403, 757], [549, 584], [238, 258], [296, 526], [586, 607], [291, 591], [62, 289], [16, 288], [581, 589], [8, 380], [655, 683], [58, 582]]
#feature_pair_plus_list = [[466, 529], [664, 759], [602, 766], [64, 665], [279, 590], [397, 592], [311, 621], [248, 755], [660, 768], [218, 666], [65, 278], [549, 607], [16, 402], [53, 757], [463, 526], [197, 312], [507, 762], [320, 619], [367, 380], [10, 350], [62, 401], [52, 756], [610, 633], [0, 656], [319, 758], [38, 50], [288, 296], [67, 584], [48, 611], [422, 724], [249, 591], [287, 295], [341, 589], [208, 728], [66, 508], [44, 605], [4, 358], [9, 695]]
#feature_pair_mul_list = [[466, 529], [621, 664], [159, 626], [599, 602], [213, 607], [209, 218], [433, 463], [16, 665], [619, 766], [158, 625], [558, 605], [64, 248], [402, 660], [583, 606], [53, 279], [595, 596], [367, 590], [592, 633], [52, 278], [65, 350], [10, 38], [526, 644], [42, 397], [23, 666], [401, 758], [67, 73], [54, 589], [507, 549], [358, 591], [423, 610], [250, 312], [311, 755], [66, 353], [611, 732], [645, 765], [1, 320], [88, 341], [319, 757], [286, 296], [375, 403], [48, 509], [203, 581], [422, 655], [62, 87], [283, 622], [627, 724], [168, 268], [0, 197], [282, 646], [420, 656]]
feature_regression = ['f2','f332','f67','f25','f120','f766','f376','f39','f670','f228','f652','f415','f596','f406','f13','f355'],

def read_data(filename):
	data0 = np.load(filename) # read the dataset
	data0_str = np.array([line.decode("utf-8") for line in data0])  # decode the dataset
	np.savetxt('temp.csv', data0_str, fmt='%s') #translate the data set to csv file
	if(filename == "ecs171train.npy"):
		data = pd.read_csv('temp.csv') # read the dataset again as a panda dataframe type
	else:
		data = pd.read_csv('temp.csv', header = None)
	data = data.fillna(0) # fill the missing data with 0
	return data

def secret_feature_select(data):  #just testing some golden features
	feature_list = []
	#for i,j in feature_pair_plus_list:
	#	feature_list.append(data[:,i]+data[:,j])
	#for i,j in feature_pair_sub_list:
	#	feature_list.append(data[:,i]-data[:,j])
	#for i,j in feature_pair_mul_list:
	#	feature_list.append(data[:,i]*data[:,j])
	
	#feature_list.append(data['f527'].values - data['f528'].values)
	feature_list.append(data['f528'].values - data['f274'].values)
	feature_list.append(data['f528'].values) #f528
	feature_list.append(data['f777'].values) #f777
	feature_list.append(data['f222'].values)
	feature_list.append(data['f68'].values)
	feature_list.append(data['f2'].values)

	feature_list = np.array(feature_list).T
	return feature_list

def reg_feature_select(data):
	feature_list = []
	feature_list.append(data['f528'].values - data['f274'].values)
	feature_list.append(data['f2'].values)
	feature_list.append(data['f332'].values)
	feature_list.append(data['f67'].values)
	feature_list.append(data['f25'].values)
	feature_list.append(data['f120'].values)
	feature_list.append(data['f766'].values)
	feature_list.append(data['f376'].values)
	feature_list.append(data['f39'].values)
	feature_list.append(data['f670'].values)
	feature_list.append(data['f228'].values)
	feature_list.append(data['f652'].values)
	feature_list.append(data['f415'].values)
	feature_list.append(data['f596'].values)
	feature_list.append(data['f406'].values)
	feature_list.append(data['f13'].values)
	feature_list.append(data['f355'].values)

	feature_list = np.array(feature_list).T
	return feature_list
	

def classification(train,test,K):
	# get the loss 
	loss = train.values[:,770].astype(int)

	# replace nonzero loss with 1
	#loss_1 = np.zeros(50000)
	
	for i in range(50000):
		if loss[i] > 0:
			loss[i] = 1
	
	X = train.values
	X_new = secret_feature_select(train)

	
	# select most important K features
	
	# X = train.values
	'''
	selector = SelectKBest(mutual_info_classif, k = K)
	selector.fit(X[:,:769], loss) 
	indexes_selected = selector.get_support(indices=True) # selected index
	print("features selected: ", indexes_selected)
	X_new = selector.transform(X[:,:769])
	'''



	#clf = SGDClassifier(loss="hinge", penalty="l2")
	#clf = tree.DecisionTreeClassifier()
	clf = GradientBoostingClassifier(n_estimators=65, learning_rate=0.3, max_depth=6, random_state=0)
	#clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
	clf.fit(X_new[:40000,:],loss[:40000]) # make first 40000 data as training and last 10000 data as validation
	print("classification score = ", clf.score(X_new[40000:,:],loss[40000:])) # get the validation score

	clf.fit(X_new,loss) # after validation, we train it with all data

	# testing for classification to get the zero terms
	Y = test.values
	test.columns = feature_index
	Y_new = secret_feature_select(test)
	
	test_data_size = Y[:,0].size  # the size of test data 
	'''
	Y_new = Y[:,indexes_selected[0]].reshape(test_data_size,1) # initialize the Y_new with the first important features 
	for col in range(1,K):
		feature = Y[:,indexes_selected[col]].reshape(test_data_size,1)
		Y_new = np.hstack((Y_new,feature))  # append the important features in test data to Y_new
	'''
	result = clf.predict(Y_new) # predict the classificaiton of the test data 
	
	nonzero = 0
	for i in range(test_data_size):
		if result[i] > 0:
			nonzero = nonzero + 1
	print("# of nonzero = ", nonzero)

	#done with classification 

	return result


def regression(train,test,clf_res,K):
	# pick up the nonzero rows from the training data
	train_nonzero = train[train.loss != 0]


	# regression feature selection

	X = train_nonzero.values
	loss = X[:,770]
	'''
	selector = SelectKBest(mutual_info_regression, k=K)
	selector.fit(X[:,:769], loss)
	indexes_selected = selector.get_support(indices=True) # selected index
	X_new = selector.transform(X[:,:769])
	'''
	
	X_new = reg_feature_select(train_nonzero)
	#regression training

	clf = linear_model.Lasso(alpha=0.1)
	# clf =   ensemble.GradientBoostingRegressor(loss='ls', learning_rate=0.0075, n_estimators=5000, subsample=0.5, min_samples_split=20, min_samples_leaf=20, max_leaf_nodes=30,random_state=9753, verbose=0)
	clf.fit(X_new[:4000,:],loss[:4000]) 
	print("regression score = ", clf.score(X_new[4000:,:],loss[4000:]))
	clf.fit(X_new,loss)

	#regression testing
	#prepare the data for testing
	test['loss'] = clf_res # add previous classfication loss result to test data
	test_nonzero = test[test.loss != 0] # remove zero rows from the test data since we are only testing nonzero data for regression

	Y = test_nonzero.values
	test_data_size = Y[:,0].size  # the size of test data 
	test.columns = feature_index2
	Y_new = reg_feature_select(test_nonzero)
	
	'''
	Y_new = Y[:,indexes_selected[0]].reshape(test_data_size,1) # initialize the Y_new with the first important features 
	for col in range(1,K):
		feature = Y[:,indexes_selected[col]].reshape(test_data_size,1)
		Y_new = np.hstack((Y_new,feature))  # append the important features in test data to Y_new
	'''
	result = clf.predict(Y_new).astype(int) # predict the classificaiton of the test data 

	# combine result
	k = 0
	for i in range(clf_res.size):
		if clf_res[i] > 0:
			clf_res[i] = result[k]
			k = k + 1

	return clf_res

if __name__ == '__main__':
	train = read_data("ecs171train.npy") # read the data for train and fill the nan  with 0
	test = read_data("ecs171test.npy") # read the data for test and fill the nan with 0
	K = 30 # features slected
  	
	clf_res = classification(train,test,K)
	reg_res = regression(train,test,clf_res,K)
	submission = pd.read_csv('ecs171sample_submission.csv')
	submission['loss'] = reg_res
	submission.to_csv('out.csv',index=False)
	np.savetxt('result.csv', reg_res, fmt='%d')

	
	sol = pd.read_csv('train_v2.csv')
	my_sol = submission
	sum = 0.0
	for i in range(my_sol['id'].size):
		sum = sum + abs(my_sol['loss'][i] - sol['loss'][my_sol['id'][i]-1])
	print("your final score is ", sum/my_sol['id'].size) 	










