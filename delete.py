import os
import shutil

name = input('请输入学号：')

people = './people'
if not os.path.exists(people):
    os.makedirs(people)

index = 1
person_all = []
for (path, dirnames, filenames) in os.walk(people):
    for filename in filenames:
        if filename.endswith('.jpg'):
            person_all.append(filename)
            index += 1

if 'person_'+name+'.jpg' in person_all:
    shutil.rmtree(name)
    shutil.rmtree('person_'+name)
    os.remove('people/person_'+name+'.jpg')
    if not os.listdir('people'):
        os.rmdir('people')
    else:
        pass
    print("删除完成！")
else:
    print("所删成员不存在，请重新输入！")



