import tensorflow as tf
import numpy as np
import cv2
import dlib
import os
import time
import sys
import random
from sklearn.model_selection import train_test_split

size = 64
people = './people'
if not os.path.exists(people):
    os.makedirs(people)
name = input("请输入学号：")
index = 1
person_all = []
for (path, dirnames, filenames) in os.walk(people):
    for filename in filenames:
        if filename.endswith('.jpg'):
            person_all.append(filename)
            index += 1

#调节照片信息
def re_img(img, light=1, bias=0):
    width = img.shape[1]
    high= img.shape[0]
    #image = []
    for i in range(0,width):
        for j in range(0,high):
            for c in range(3):
                tmp = int(img[j,i,c]*light + bias)
                if tmp > 255:
                    tmp = 255
                elif tmp < 0:
                    tmp = 0
                img[j,i,c] = tmp
    return img

def get_pface_img(p,path):
    #人脸识别器
    detector = dlib.get_frontal_face_detector()
    #调用摄像头
    camera = cv2.VideoCapture(0)
    #从摄像头中获取要识别的人脸图片
    index = 1
    while True:
        #拍200张图片
        if (index <= 100):
            # 从摄像头读取照片
            success, img = camera.read()
            # 转为灰度图片
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 使用detector进行人脸检测
            dets = detector(gray_img, 1)
            #识别人脸图片
            for i, d in enumerate(dets):
                x1 = d.top() if d.top() > 0 else 0
                y1 = d.bottom() if d.bottom() > 0 else 0
                x2 = d.left() if d.left() > 0 else 0
                y2 = d.right() if d.right() > 0 else 0
                face = img[x1:y1,x2:y2]
                print('take pictuer',index)
                # 数据集增强，随机调整图片的对比度与亮度
                face = re_img(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
                face = cv2.resize(face, (size,size))
                #显示捕捉到得人脸图片
                cv2.imshow('image', face)
                #将捕捉到的人脸图片保存到文件夹中
                cv2.imwrite(path+'/person_'+str(p)+'_'+str(index)+'.jpg', face)
                index += 1
            key = cv2.waitKey(30) & 0xff
            if key == 27:
                break
        else:
            #释放摄像头
            camera.release()
            #关闭图像窗口
            cv2.destroyAllWindows()
            #返回结束信息
            print('get face_img OK,thank you!')
            break

def getPaddingSize(img):
    h, w, _ = img.shape
    top, bottom, left, right = (0,0,0,0)
    longest = max(h, w)
    if w < longest:
        tmp = longest - w
        left = tmp // 2
        right = tmp - left
    elif h < longest:
        tmp = longest - h
        top = tmp // 2
        bottom = tmp - top
    else:
        pass
    return top, bottom, left, right
#涂黑比例外部分并截取人脸图片
def readData(path , h=size, w=size):
    imgs_re = []
    labs_re = []
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            filename = path + '/' + filename
            img = cv2.imread(filename)
            top,bottom,left,right = getPaddingSize(img)
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
            img = cv2.resize(img, (h, w))
            #将调整后的图片存入imgs中
            imgs_re.append(img)
            #将对应的标签存入labs中
            labs_re.append(path)
    print("图片redata成功")
    return imgs_re,labs_re



def people_login():
    if 'person_'+name+'.jpg' in person_all:
        path = 'person_'+ name
    else:
        print("start login a new person")
        path = 'person_' + name
        if not os.path.exists(path):
            os.makedirs(path)
        print("start take face_img")
        get_pface_img(name, path)
    other_face = './other_faces'
    img_a,lab_a = readData(path)
    imgs = img_a
    labs = lab_a
    img_b,lab_b = readData(other_face)
    imgs = imgs + img_b
    labs = labs + lab_b
    return imgs,labs

path = 'person_' + name
imgs,labs = people_login()
imgs = np.array(imgs)
labs = np.array([[0,1] if lab == path else [1,0] for lab in labs])
# 随机划分测试集与训练集
train_x,test_x,train_y,test_y = train_test_split(imgs, labs, test_size=0.05, random_state=random.randint(0,100))
# 参数：图片数据的总数，图片的高、宽、通道
train_x = train_x.reshape(train_x.shape[0], size, size, 3)
test_x = test_x.reshape(test_x.shape[0], size, size, 3)
# 将数据转换成小于1的数
train_x = train_x.astype('float32')/255.0
test_x = test_x.astype('float32')/255.0

print('train size:%s, test size:%s' % (len(train_x), len(test_x)))
# 图片块，每次取100张图片
batch_size = 100
num_batch = len(train_x) // batch_size

x = tf.placeholder(tf.float32, [None, size, size, 3])
y_ = tf.placeholder(tf.float32, [None, 2])

keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)

def weightVariable(shape):
    init = tf.random_normal(shape, stddev=0.01)
    return tf.Variable(init)

def biasVariable(shape):
    init = tf.random_normal(shape)
    return tf.Variable(init)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxPool(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def dropout(x, keep):
    return tf.nn.dropout(x, keep)

def cnnLayer():
    # 第一层
    W1 = weightVariable([3,3,3,32]) # 卷积核大小(3,3)， 输入通道(3)， 输出通道(32)
    b1 = biasVariable([32])
    # 卷积
    conv1 = tf.nn.relu(conv2d(x, W1) + b1)
    # 池化
    pool1 = maxPool(conv1)
    # 减少过拟合，随机让某些权重不更新
    drop1 = dropout(pool1, keep_prob_5)

    # 第二层
    W2 = weightVariable([3,3,32,64])
    b2 = biasVariable([64])
    conv2 = tf.nn.relu(conv2d(drop1, W2) + b2)
    pool2 = maxPool(conv2)
    drop2 = dropout(pool2, keep_prob_5)

    # 第三层
    W3 = weightVariable([3,3,64,64])
    b3 = biasVariable([64])
    conv3 = tf.nn.relu(conv2d(drop2, W3) + b3)
    pool3 = maxPool(conv3)
    drop3 = dropout(pool3, keep_prob_5)

    # 全连接层
    Wf = weightVariable([8*16*32, 512])
    bf = biasVariable([512])
    drop3_flat = tf.reshape(drop3, [-1, 8*16*32])
    dense = tf.nn.relu(tf.matmul(drop3_flat, Wf) + bf)
    dropf = dropout(dense, keep_prob_75)

    # 输出层
    Wout = weightVariable([512,2])
    bout = weightVariable([2])
    #out = tf.matmul(dropf, Wout) + bout
    out = tf.add(tf.matmul(dropf, Wout), bout)
    return out


output = cnnLayer()

if 'person_'+name+'.jpg' not in person_all :
    out = cnnLayer()

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y_))

    train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

    # 比较标签是否相等，再求的所有数的平均值，tf.cast(强制转换类型)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(y_, 1)), tf.float32))

    # 将loss与accuracy保存以供tensorboard使用
    tf.summary.scalar('loss', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()

    # 数据保存器的初始化
    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter(name+'/tmp', graph=tf.get_default_graph())

        for n in range(10):
             # 每次取128(batch_size)张图片
            for i in range(num_batch):
                batch_x = train_x[i*batch_size : (i+1)*batch_size]
                batch_y = train_y[i*batch_size : (i+1)*batch_size]
                # 开始训练数据，同时训练三个变量，返回三个数据
                _,loss,summary = sess.run([train_step, cross_entropy, merged_summary_op],
                                           feed_dict={x:batch_x,y_:batch_y, keep_prob_5:0.5,keep_prob_75:0.75})
                summary_writer.add_summary(summary, n*num_batch+i)
                # 打印损失
                print(n*num_batch+i, loss)

                if (n*num_batch+i) % 100 == 0:
                    # 获取测试数据的准确率
                    acc = accuracy.eval({x:test_x, y_:test_y, keep_prob_5:1.0, keep_prob_75:1.0})
                    print(n*num_batch+i, acc)
                    # 准确率大于0.98时保存并退出
                    if (acc > 0.98 and n > 2) or n*num_batch+i > 700:
                        saver.save(sess, name+'/model.ckpt')
                        break
            if (acc > 0.98 and n > 2) or n*num_batch+i > 700:
                break
        print('accuracy less 0.98, exited!')
    sess.close()

    login_img = cv2.imread('./person_'+name+'/person_' + name + '_1.jpg')
    cv2.imwrite('people./person_' + name + '.jpg', login_img)

else:
    pass

start = time.clock()

predict = tf.argmax(output, 1)

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, './'+name+'/model.ckpt')

def is_my_face(image):
    res = sess.run(predict, feed_dict={x: [image/255.0], keep_prob_5:1.0, keep_prob_75: 1.0})
    if res[0] == 1:
        return True
    else:
        return False

#使用dlib自带的frontal_face_detector作为我们的特征提取器
detector = dlib.get_frontal_face_detector()

cam = cv2.VideoCapture(0)

while True:
    _, img = cam.read()
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray_image, 1)
    if not len(dets):
        #print('Can`t get face.')
        cv2.imshow('img', img)
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break

    for i, d in enumerate(dets):
        end = time.clock()
        if int(end - start) < 10:
            x1 = d.top() if d.top() > 0 else 0
            y1 = d.bottom() if d.bottom() > 0 else 0
            x2 = d.left() if d.left() > 0 else 0
            y2 = d.right() if d.right() > 0 else 0
            face = img[x1:y1,x2:y2]
            # 调整图片的尺寸
            face = cv2.resize(face, (size,size))
            print('Is this my face? %s' % is_my_face(face))

            cv2.rectangle(img, (x2,x1),(y2,y1), (255,0,0),3)
            cv2.imshow('image',img)
            key = cv2.waitKey(30) & 0xff
            if key == 27:
                break
        else:
            sys.exit(0)


sess.close()

