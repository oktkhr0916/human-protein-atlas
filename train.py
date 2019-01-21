import pandas     as pd
import numpy      as np
import tensorflow as tf
import cv2
import math
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random


class CNN:

    def __init__(self):
        """
        """


    def cmyk_array_unify(ary_c, ary_m, ary_y, ary_k):
        """
        
        """
        # 一次元化して配列が同じか。
        len_c = len(ary_c.reshape(-1,))
        len_m = len(ary_m.reshape(-1,))
        len_y = len(ary_y.reshape(-1,))
        len_k = len(ary_k.reshape(-1,))

        
        if( len_c - len_m + len_y - len_k ) == 0 :
            cmyk = []
            
            d_y_2 = len(ary_c)
            d_x_2 = len(ary_c[0])
            
            for i in range(d_y_2):
                d_x_3 = []
                for j in range(d_x_2):
                    d_z_3 = []
                    d_z_3.append(ary_c[i][j])
                    d_z_3.append(ary_m[i][j])
                    d_z_3.append(ary_y[i][j])
                    d_z_3.append(ary_k[i][j])
                    d_x_3.append(d_z_3)
                cmyk.append(d_x_3)
            return np.array(cmyk)
        
        else:
            return 0

    def hpaic_image_loader(img_name, folder="./data/train/", resize=(128,128), dim_1 = True):
        """
        image loading and unifying as 3-level tensor
        """
        img_blue   = cv2.resize(cv2.imread( folder + img_name + "_blue"   + ".png", 0), dsize = resize)
        img_green  = cv2.resize(cv2.imread( folder + img_name + "_green"  + ".png", 0), dsize = resize)
        img_red    = cv2.resize(cv2.imread( folder + img_name + "_red"    + ".png", 0), dsize = resize)
        img_yellow = cv2.resize(cv2.imread( folder + img_name + "_yellow" + ".png", 0), dsize = resize)
    
        if dim_1:
            return cmyk_array_unify(img_blue, img_green, img_red, img_yellow).reshape(-1,)
        
        return cmyk_array_unify(img_blue, img_green, img_red, img_yellow)

    def o_train_test_split(x_array, y_array, ratio=0.7):
        """
        train_data_split
        """
        x_train_len         = int( len(x_array) * ratio )
        x_train_index_array = random.sample(range(0, len(x_array), 1), k = x_train_len)
        x_test_index_array  = list(set(range(0, len(x_array), 1)) - set(x_train_index_array))
        x_train_array       = [x_array[i] for i in x_train_index_array]
        x_test_array        = [x_array[i] for i in x_test_index_array]
        y_train_array       = [y_array[i] for i in x_train_index_array]
        y_test_array        = [y_array[i] for i in x_test_index_array]
        
        return (x_train_array, x_test_array, y_train_array, y_test_array)

    def weight_variable(self, shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    
    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2,2, 1],
                            strides=[1,2,2, 1], padding='SAME')

    def inference(self, x, keep_prob):
        """
        第1層  (バッチ正規化層)
        第2層  (畳み込み層)
        第3層  (バッチ正規化層)
        第4層  (畳み込み層)
        第5層  (バッチ正規化層)
        第6層  (畳み込み層)
        第7層  (バッチ正規化層)
        第8層  (プーリング層)
        第9層  (畳み込み層)
        第10層 (バッチ正規化層)
        第11層 (プーリング層)
        第12層 (畳み込み層)
        第13層 (バッチ正規化層)
        第14層 (プーリング層)
        第15層 (畳み込み層)
        第16層 (バッチ正規化層)
        第17層 (プーリング層)
        第18層 (畳み込み層)
        第19層 (バッチ正規化層)
        第20層 (プーリング層)
        第21層 (平坦化層)
        第22層 (全結合層)
        第23層 (全結合層)
        """
        x_image = tf.reshape(x, [-1, 128, 128, 4])

        y_norm1 = tf.contrib.layers.batch_norm(x_image)

        W_conv1 = self.weight_variable([3, 3, 4, 8])
        b_conv1 = self.bias_variable([8])
        y_conv1 = tf.nn.relu(conv2d(y_norm1, W_conv1) + b_conv1)

        y_norm2 = tf.contrib.layers.batch_norm(y_conv1)

        W_conv2 = self.weight_variable([3, 3, 8, 8])
        b_conv2 = self.bias_variable([8])
        y_conv2 = tf.nn.relu(conv2d(y_norm2, W_conv2) + b_conv2)

        y_norm3 = tf.contrib.layers.batch_norm(y_conv2)

        W_conv3 = self.weight_variable([3, 3, 8, 16])
        b_conv3 = self.bias_variable([16])
        y_conv3 = tf.nn.relu(conv2d(y_norm3, W_conv3) + b_conv3)

        y_norm4 = tf.contrib.layers.batch_norm(y_conv3)

        y_pool1 = self.max_pool_2x2(y_norm4) #64*64
        y_drop1 = tf.nn.dropout(y_pool1, keep_prob)

        W_conv4 = self.weight_variable([3, 3, 16, 32])
        b_conv4 = self.bias_variable([32])
        y_conv4 = tf.nn.relu(conv2d(y_drop1, W_conv4) + b_conv4)

        y_norm5 = tf.contrib.layers.batch_norm(y_conv4)

        y_pool2 = self.max_pool_2x2(y_norm5) #32*32
        y_drop2 = tf.nn.dropout(y_pool2, keep_prob)

        W_conv5 = self.weight_variable([3, 3, 32, 64])
        b_conv5 = self.bias_variable([64])
        y_conv5 = tf.nn.relu(conv2d(y_drop2, W_conv5) + b_conv5)

        y_norm6 = tf.contrib.layers.batch_norm(y_conv5)

        y_pool3 = self.max_pool_2x2(y_norm6) #16*16
        y_drop3 = tf.nn.dropout(y_pool3, keep_prob)

        W_conv6 = self.weight_variable([3, 3, 64, 128])
        b_conv6 = self.bias_variable([128])
        y_conv6 = tf.nn.relu(conv2d(y_drop3, W_conv6) + b_conv6)

        y_norm7 = tf.contrib.layers.batch_norm(y_conv6)

        y_pool4 = self.max_pool_2x2(y_norm7) #8*8
        y_drop4 = tf.nn.dropout(y_pool4, keep_prob)

        W_conv7 = self.weight_variable([3, 3, 128, 256])
        b_conv7 = self.bias_variable([256])
        y_conv7 = tf.nn.relu(conv2d(y_drop4, W_conv7) + b_conv7)

        y_norm8 = tf.contrib.layers.batch_norm(y_conv7)

        y_pool5 = self.max_pool_2x2(y_norm8) #4*4
        y_drop5 = tf.nn.dropout(y_pool5, keep_prob)

        y_pool2_flat = tf.reshape(y_drop5, [-1, 4096])
        
        W_fc1 = self.weight_variable([4096, 1024])
        b_fc1 = self.bias_variable([1024])
        y_fc1 = tf.nn.relu(tf.matmul(y_pool2_flat, W_fc1) + b_fc1)
        
        W_fc2 = self.weight_variable([1024, 28])
        b_fc2 = self.bias_variable([28])

        y = tf.sigmoid(tf.matmul(y_fc1, W_fc2) + b_fc2) 

        return y


    def loss(y, t):
        """
        
        """
        cross_entropy = -tf.reduce_sum( t * tf.log(y + 1e-9)) + ((1-t) * tf.log(1 - y + 1e-9) )
        return cross_entropy
            

    def training(loss):
        """
        学習アルゴリズムの定義
        """
        optimizer   = tf.train.AdamOptimizer(0.01)
        train_setep = optimizer.minimize(loss) 

    def accuracy(self, y, t):
        """
        F value (But I couldn't make this.)
        """
        return accuracy

    def fit(self, X_train, Y_train, epoch, batch_size):
        """
        """
        keep_prob = tf.placeholder("float")
        x         = tf.placeholder("float", [None, 65536])
        t         = tf.placeholder("float", [None, 28])

        # for evaluation
        self.epoch      = epoch
        self.batch_size = batch_size
        self._x         = x
        self._t         = t
        self.keep_prob  = keep_prob

        y          = self.inference(x, keep_prob)
        loss       = self.loss(y, t)
        train_step = self.training(loss)
        accuracy   = self.accuracy(y, t)

        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()

        x_train_name_len  = len(X_train)
        batch_count       = int(x_train_name_len / self.batch_size) # 21750 / 50 = 435
        batch_count_mod   = x_train_name_len % self.batch_size
        b_top_index_array = random.sample(range(
                                            0, 
                                            x_train_name_len, 
                                            self.batch_size), 
                                        k = self.batch_size - 1 )

        with tf.Session() as sess:
        
            sess.run(init_g)
            sess.run(init_l)

            for i in tqdm(range(self.epoch)):
                for p_index in tqdm(b_top_index_array):
                    # If last batch
                    if p_index == batch_count * batch_size:
                        index_array = list(range(p_index, p_index + batch_count_mod ))
                    else:
                        index_array = list(range(p_index, p_index + batch_size))
                    
                    patch_x = [hpaic_image_loader(X_train[i]) for i in index_array]
                    patch_t = [y_train_array[i] for i in index_array]
                    
                    sess.run(train_step, feed_dict={x: patch_x, t: patch_t, keep_prob: 0.5})

            saver = tf.train.Saver()
            saver.save(sess, "./model/hpaic_model")

    def predict(self):
        """
        """
        sess = tf.Session()
        sess.run(init_g)
        sess.run(init_l)
        saver = tf.train.import_meta_graph('./model/hpaic_model.meta')
        saver.restore(sess,  tf.train.latest_checkpoint('./model/'))

        sub_df = pd.read_csv("./data/sample_submission.csv")

        size        = 500
        length      = len(sub_image_name)
        count       = int(length / size) # 21750 / 500 = 43
        count_mod   = length % size      # 21750 % 500 = 250

        pred = []
        for i in range(count):
            s_index = i*size
            e_index = i*size+ size
            
            if i == count -1:
                e_index = e_index + count_mod
            
            sub_image_array = np.array([hpaic_image_loader(j, "./data/test/" ) for j in tqdm(sub_image_name[s_index : e_index])])
            pred.extend(sess.run(y, feed_dict={x: sub_image_array, keep_prob: 1.0}))

        a = [[1 if j>=0.5 else 0 for  j in i]for i in pred]
        index_array = [[ str(j) if i[j] else " "  for j in range(len(i)) ]for i in a]
        string_array = [[ j for j in i if j != " "]for i in index_array]
        join_array = [' '.join(i) for i in string_array]
        submit = pd.DataFrame({"Id":sub_image_name, "Predicted":join_array})

        submit.to_csv("submit/submit.csv" , columns=['Id',"Predicted" ], index=False)

        
if __name__ == '__main__':
    """
    
    """
    # yの読み出し（最初の9個）
    train_df = pd.read_csv("./data/train.csv")
    # image_id
    image_name = np.array(train_df["Id"][:])

    # target_vector
    t_data = np.array(train_df.iloc[:,1:])

    X_train_name_array, X_test_name_array, y_train_array, y_test_array = o_train_test_split(image_name, t_data)

    cnn = CNN()
    cnn.fit(X_train_name_array, y_train_array, 20, 500)


