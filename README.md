# DRQN-tensorflow
Play grid_world with DRQN.

环境：
Python 3.6
tensorflow 2.0

2019.12.5————————————————————————————————  

从这里找到原版代码https://github.com/WangChen100/DQNs_Gridworld  

主要信息————————————  

1、输入是[1,84,84,3]大小的图片batchsize:4，转化成[?,21168]大小的向量，再reshape成[?,84,84,3]进入四层卷积，经过单元数为512的lstm得到当前隐状态ht,以2:1比例分为advantage 和 value值，计算最终的Qout。

2、targetQ和MainQ用的是同一个Q网络。每5steps用主网络更新目标网络，更新lstm隐藏状态：  
                if total_steps % (update_freq) == 0:  
                    updateTarget(targetOps, sess)  
                    # Reset the recurrent layer's hidden state  
                    ###################################################更新隐藏状态  
                    state_train = (np.zeros([batch_size, h_size]), np.zeros([batch_size, h_size]))  


3、损失函数为target和Qout的平方误差，计算损失的时，用mask去掉了一半的值（现在还不知道原因）

主要修改——————————

1、去掉了gif的存储，因为存了一堆看不明白的图

2、输出每10次迭代的loss图  

后期计划（目标是用RNN和RL做路径规划）———————  

1、修改输入为一张黑白的障碍物地图，输出离散动作以到达目的地  
2、修改网络框架，输出连续动作值，参考DDPG  
3、DDPG、AC和Lstm的尝试结合  
4、多智能体的路径规划  


![image](https://github.com/Elly0723/DRQN-tensorflow/blob/master/images/Screenshot%20from%202019-12-05%2011-17-06.png)


![image](https://github.com/Elly0723/DRQN-tensorflow/blob/master/images/Screenshot%20from%202019-12-05%2011-22-03.png)
