''' This script was needed to put hmdb51 dataset
train-test splits into the same format of the UCF-101 splits'''
import os
cwd = os.getcwd()
data_dir = os.path.join(cwd,'data')
list_dir = os.path.join(data_dir,'hmdb51_test_train_splits')
files = os.listdir(list_dir)
test_splits = open(os.path.join(list_dir,'testlist.txt'),'w')
train_splits = open(os.path.join(list_dir,'trainlist.txt'),'w')
classes = []
j=0
for file in files:
    if j%3 == 0:
        cat = file[0:file.find('test')-1]
        if not cat in classes:
            classes.append(cat)
        read = open(os.path.join(list_dir,file),'r')
        for line in read.readlines():
            if int(line[-3]) == 2:
                test_splits.write(cat + '/' + line[0:-3]+'\n')
            else:
                train_splits.write(cat+'/'+line[0:-3]+'\n')
        read.close()
    j+=1
test_splits.close()
train_splits.close()
class_index = open(os.path.join(list_dir,'classInd.txt'),'w')
i = 1
for clase in classes:
    class_index.write(str(i)+' '+clase+'\n')
    i+=1
class_index.close()
