import pandas as pd  # for data manipulation
from numpy import *
import plotly.express as px  # for data visualization
import umap
from clustering.bi_kmeans import *
import utils

device = get_device()
# ------------------ set up CodeFormer restorer -------------------
vqgan = ARCH_REGISTRY.get('VQAutoEncoder')(img_size=512, nf=64, ch_mult=[1, 2, 2, 4, 4, 8],
                                           quantizer='nearest', res_blocks=2, attn_resolutions=[16],
                                           codebook_size=1024).to(device)

ckpt_path = '/opt/data/liyan/EDICT/weights/vqgan_code1024.pth'
# ckpt_path = 'C:\\Users\\97860\\exp\\EDICT\\weights\\vqgan_code1024.pth'
checkpoint = torch.load(ckpt_path)['params_ema']
vqgan.load_state_dict(checkpoint)
vqgan.eval()

X = vqgan.quantize.embedding.weight.data.cpu().numpy()
y = np.zeros(1024)



class Rubust_Codebook(object):
    """
    @brief  把鲁棒性codebook写成类，方便使用
    @args   codebook_data:      外部指定的VQGAN的codebook参数
    @args   dimension:          数据维度：256
    @args   code_len:           需要编码的码字长度，如数据长度为1024，则码字长度为10 (1024 = 2^10)
    """

    def __init__(self, codebook_data, code_len, y, y_center):
        self.data_mat = mat(codebook_data.tolist())  # 替代data_mat = mat(vqgan.quantize.embedding.weight.tolist())
        self.bin_code_list = [['0000000000', False]] * self.data_mat.shape[0]  # 编码存储数组，初始化为全0，确定编码后第二个值为True
        # self.data_dimension = dimension  # 数据维度
        self.code_len = code_len  # 码字长度
        self.y = y  # 可视化label
        self.y_center = y_center

    def encode_func_random(self, center_code, assment, data_father, k_next, flip):
        """
        @param      center_code     上次聚类结果中心的编码列表，根据每一类父簇进行随机翻转
        @param      assment     上次聚类结果，用来筛选父簇中某一子簇的数据，进行编码/进一步聚类+编码
        @param      data_father  父簇数据
        @param      k_next      下一层kmeans所需要分的类
        @param      flip        某轮的翻转策略，设计为第一轮C6_2=15，第二轮则为6
        """
        for i in range(len(center_code)):
            # 第i类子簇
            data_filtered = filter_data_by_cluster(assment, data_father, i)

            # 如果子簇数量<=10，停止聚类（否则空簇报错），对每个数据直接编码（size是为了方便判断，2是数据维度）
            if data_filtered.shape[0] <= self.code_len:
                for j in range(data_filtered.shape[0]):
                    sub_data = data_filtered[j]
                    sub_data_index = np.where(np.all(self.data_mat == sub_data, axis=1))[0][0]

                    # 如果未进行编码，则调用随机翻转算法进行编码
                    if not self.bin_code_list[sub_data_index][1]:
                        gen_code = find_unique_flipped_sample(self.bin_code_list, center_code[i], flip=flip, n=1)[0]
                        self.bin_code_list[sub_data_index] = [gen_code, True]
                    # 否则就不用处理，直接进行下一个sub_data的处理

            # 如果子簇数量>10，则对每个中心最近的数据进行编码，对每个进一步进行kmeans
            else:
                centroid1, assment1 = bi_kMeans(data_filtered, k_next)  # 第一次聚类的k是指定的

                new_center_code = []  # 存储新的father编码，供下一轮递归使用
                # 根据子聚类结果的中心，进行编码
                min_distance_indices = find_min_distance_indices(assment1)
                for center in min_distance_indices:
                    sub_data = data_filtered[center]
                    sub_data_index = np.where(np.all(self.data_mat == sub_data, axis=1))[0][0]

                    # 如果未进行编码，则调用随机翻转算法进行编码
                    if not self.bin_code_list[sub_data_index][1]:
                        gen_code = find_unique_flipped_sample(self.bin_code_list, center_code[i], flip=flip, n=1)[0]
                        self.bin_code_list[sub_data_index] = [gen_code, True]
                        new_center_code.append(gen_code)
                    # 否则就不用处理，直接进行下一个sub_data的处理

                # 编码完本簇的中心，仍需对子簇的子簇进行递归的编码，此时，子簇data_filtered作为父簇
                self.encode_func_random(new_center_code, assment1, data_filtered, self.code_len, 1)

    # 原版第二轮开始只翻转一位，现在试试第二轮翻转2位
    def encode_func_random_with_round(self, center_code, assment, data_father, k_next, round):
        """
        @param      center_code     上次聚类结果中心的编码列表，根据每一类父簇进行随机翻转
        @param      assment     上次聚类结果，用来筛选父簇中某一子簇的数据，进行编码/进一步聚类+编码
        @param      data_father  父簇数据
        @param      k_next      下一层kmeans所需要分的类
        @param      round       根据聚类轮数，设置某轮的翻转位数flip，翻转多少bit
        """
        if round == 1:
            flip = 2
        elif round == 2:
            flip = 1
        else:
            flip = 1

        print("round = ", round)

        for i in range(len(center_code)):
            # 第i类子簇
            data_filtered = filter_data_by_cluster(assment, data_father, i)

            # 如果子簇数量<=10，停止聚类（否则空簇报错），对每个数据直接编码（size是为了方便判断，2是数据维度）
            if data_filtered.shape[0] <= self.code_len:
                for j in range(data_filtered.shape[0]):
                    sub_data = data_filtered[j]
                    sub_data_index = np.where(np.all(self.data_mat == sub_data, axis=1))[0][0]

                    # 如果未进行编码，则调用随机翻转算法进行编码
                    if not self.bin_code_list[sub_data_index][1]:
                        gen_code = find_unique_flipped_sample(self.bin_code_list, center_code[i], flip=flip, n=1)[0]
                        self.bin_code_list[sub_data_index] = [gen_code, True]
                    # 否则就不用处理，直接进行下一个sub_data的处理

            # 如果子簇数量>10，则对每个中心最近的数据进行编码，对每个进一步进行kmeans
            else:
                centroid1, assment1 = bi_kMeans(data_filtered, k_next)  # 第一次聚类的k是指定的

                new_center_code = []  # 存储新的father编码，供下一轮递归使用
                # 根据子聚类结果的中心，进行编码
                min_distance_indices = find_min_distance_indices(assment1)
                for center in min_distance_indices:
                    sub_data = data_filtered[center]
                    sub_data_index = np.where(np.all(self.data_mat == sub_data, axis=1))[0][0]

                    # 如果未进行编码，则调用随机翻转算法进行编码
                    if not self.bin_code_list[sub_data_index][1]:
                        gen_code = find_unique_flipped_sample(self.bin_code_list, center_code[i], flip=flip, n=1)[0]
                        self.bin_code_list[sub_data_index] = [gen_code, True]
                        new_center_code.append(gen_code)
                    # 否则就不用处理，直接进行下一个sub_data的处理

                # 编码完本簇的中心，仍需对子簇的子簇进行递归的编码，此时，子簇data_filtered作为父簇
                self.encode_func_random_with_round(new_center_code, assment1, data_filtered, self.code_len, round=round+1)

    def encode_func_random1(self, center_code, assment, data_father, k_next, flip):
        """
        @param      center_code     上次聚类结果中心的编码列表，根据每一类父簇进行随机翻转
        @param      assment     上次聚类结果，用来筛选父簇中某一子簇的数据，进行编码/进一步聚类+编码
        @param      data_father  父簇数据
        @param      k_next      下一层kmeans所需要分的类
        @param      flip        某轮的翻转策略，设计为第一轮C6_2=15，第二轮则为6
        """
        for i in range(len(center_code)):
            # 第i类子簇
            data_filtered = filter_data_by_cluster(assment, data_father, i)

            # 如果子簇数量<=10，停止聚类（否则空簇报错），对每个数据直接编码（size是为了方便判断，2是数据维度）
            if data_filtered.shape[0] <= self.code_len:
                sub_data_index_list = []  # 子簇内部中心在源数据的索引
                for j in range(data_filtered.shape[0]):
                    sub_data = data_filtered[j]
                    sub_data_index = np.where(np.all(self.data_mat == sub_data, axis=1))[0][0]
                    sub_data_index_list.append(sub_data_index)

                gen_code = find_unique_flipped_sample(self.bin_code_list, center_code[i], flip=flip, n=len(sub_data_index_list))
                j1 = 0
                for sub_data_index in sub_data_index_list:
                    # 如果未进行编码，则调用随机翻转算法进行编码
                    if not self.bin_code_list[sub_data_index][1]:
                        self.bin_code_list[sub_data_index] = [gen_code[j1], True]
                    j1 += 1

            # 如果子簇数量>10，则对每个中心最近的数据进行编码，对每个进一步进行kmeans
            else:
                centroid1, assment1 = bi_kMeans(data_filtered, k_next)  # 第一次聚类的k是指定的

                new_center_code = []  # 存储新的father编码，供下一轮递归使用
                # 根据子聚类结果的中心，进行编码
                min_distance_indices = find_min_distance_indices(assment1)
                sub_data_index_list = []  # 子簇内部中心在源数据的索引
                for center in min_distance_indices:
                    sub_data = data_filtered[center]
                    sub_data_index = np.where(np.all(self.data_mat == sub_data, axis=1))[0][0]
                    sub_data_index_list.append(sub_data_index)

                gen_code = find_unique_flipped_sample(self.bin_code_list, center_code[i], flip=flip, n=len(sub_data_index_list))
                j1 = 0
                for sub_data_index in sub_data_index_list:
                    # 如果未进行编码，则调用随机翻转算法进行编码
                    if not self.bin_code_list[sub_data_index][1]:
                        self.bin_code_list[sub_data_index] = [gen_code[j1], True]
                        new_center_code.append(gen_code[j1])
                    j1 += 1

                # 编码完本簇的中心，仍需对子簇的子簇进行递归的编码，此时，子簇data_filtered作为父簇
                self.encode_func_random1(new_center_code, assment1, data_filtered, self.code_len, 1)

    def encode_func_random1_with_round(self, center_code, assment, data_father, k_next, round):
        """
        @param      center_code     上次聚类结果中心的编码列表，根据每一类父簇进行随机翻转
        @param      assment     上次聚类结果，用来筛选父簇中某一子簇的数据，进行编码/进一步聚类+编码
        @param      data_father  父簇数据
        @param      k_next      下一层kmeans所需要分的类
        @param      flip        某轮的翻转策略，设计为第一轮C6_2=15，第二轮则为6
        """
        if round == 1:
            flip = 4
        elif round == 2:
            flip = 2
        else:
            flip = 1

        print("round = ", round)

        for i in range(len(center_code)):
            # 第i类子簇
            data_filtered = filter_data_by_cluster(assment, data_father, i)

            # 如果子簇数量<=10，停止聚类（否则空簇报错），对每个数据直接编码（size是为了方便判断，2是数据维度）
            if data_filtered.shape[0] <= self.code_len:
                sub_data_index_list = []  # 子簇内部中心在源数据的索引
                for j in range(data_filtered.shape[0]):
                    sub_data = data_filtered[j]
                    sub_data_index = np.where(np.all(self.data_mat == sub_data, axis=1))[0][0]
                    sub_data_index_list.append(sub_data_index)

                gen_code = find_unique_flipped_sample(self.bin_code_list, center_code[i], flip=flip, n=len(sub_data_index_list))
                j1 = 0
                for sub_data_index in sub_data_index_list:
                    # 如果未进行编码，则调用随机翻转算法进行编码
                    if not self.bin_code_list[sub_data_index][1]:
                        self.bin_code_list[sub_data_index] = [gen_code[j1], True]
                    j1 += 1

            # 如果子簇数量>10，则对每个中心最近的数据进行编码，对每个进一步进行kmeans
            else:
                centroid1, assment1 = bi_kMeans(data_filtered, k_next)  # 第一次聚类的k是指定的

                new_center_code = []  # 存储新的father编码，供下一轮递归使用
                # 根据子聚类结果的中心，进行编码
                min_distance_indices = find_min_distance_indices(assment1)
                sub_data_index_list = []  # 子簇内部中心在源数据的索引
                for center in min_distance_indices:
                    sub_data = data_filtered[center]
                    sub_data_index = np.where(np.all(self.data_mat == sub_data, axis=1))[0][0]
                    sub_data_index_list.append(sub_data_index)

                gen_code = find_unique_flipped_sample(self.bin_code_list, center_code[i], flip=flip, n=len(sub_data_index_list))
                j1 = 0
                for sub_data_index in sub_data_index_list:
                    # 如果未进行编码，则调用随机翻转算法进行编码
                    if not self.bin_code_list[sub_data_index][1]:
                        self.bin_code_list[sub_data_index] = [gen_code[j1], True]
                        new_center_code.append(gen_code[j1])
                    j1 += 1

                # 编码完本簇的中心，仍需对子簇的子簇进行递归的编码，此时，子簇data_filtered作为父簇
                self.encode_func_random1_with_round(new_center_code, assment1, data_filtered, self.code_len, round=round+1)

    # 改进的编码方案：不要随机分配码字，挑选出N个码字后，考虑与父亲兄弟中心的特征距离，来分配码字
    def encode_func_sorted(self, center_code, center_index, assment, data_father, k_next, flip):
        """
        @param      center_code      上次聚类结果中心的编码列表，根据每一类父簇进行随机翻转
        @param      center_index     上次聚类结果中心对应向量在源数据中的index
        @param      assment     上次聚类结果，用来筛选父簇中某一子簇的数据，进行编码/进一步聚类+编码
        @param      data_father  父簇数据
        @param      k_next      下一层kmeans所需要分的类
        @param      flip        某轮的翻转策略，设计为第一轮C6_2=15，第二轮则为6
        """
        for i in range(len(center_code)):
            # 第i类子簇
            data_filtered = filter_data_by_cluster(assment, data_father, i)

            # 如果子簇数量<=10，停止聚类（否则空簇报错），对每个数据直接编码（size是为了方便判断，2是数据维度）
            if data_filtered.shape[0] <= self.code_len:
                sub_data_index_list = []  # 子簇内部中心在源数据的索引
                for j in range(data_filtered.shape[0]):
                    sub_data = data_filtered[j]
                    sub_data_index = np.where(np.all(self.data_mat == sub_data, axis=1))[0][0]
                    sub_data_index_list.append(sub_data_index)

                # ============step1：根据特征距离，为后面选择汉明距离做参考====================
                # 计算子聚类中心和各个父亲兄弟中心的特征距离
                feature_distance_table = generate_feature_distance_table(self.data_mat, sub_data_index_list, center_index, i)

                # 得到各个兄弟的特征距离后，取其中距离最近的元素，进行特征距离的排序，以指导汉明距离选择
                min_bro, min_feature_dis = find_min_bro(feature_distance_table)  # 记录是从哪个父亲兄弟获得的最小特征距离，以后分配码字考虑该父亲兄弟的汉明距离

                # 对最小特征距离表进行排序，返回排序后原特征距离的表的索引，用于指导汉明距离选择更近的
                sorted_feature_dis_indice = get_sorted_indices(min_feature_dis)

                # ============step2：根据特征距离，选择汉明距离编码====================
                # 根据父亲i进行翻转策略产生候选编码
                gen_code = find_unique_flipped_sample(self.bin_code_list, center_code[i], flip=flip, n=len(sorted_feature_dis_indice))

                # 计算子聚类中心候选编码和各个父亲兄弟中心的汉明距离
                ham_distance_table = generate_ham_distance_table(self.bin_code_list, gen_code, center_index, i)

                # ============step3：分配汉明编码====================
                # 对每个父亲兄弟中心的汉明距离表，返回排序后原特征距离的表的索引
                sorted_ham_all = []
                for l in ham_distance_table:
                    sorted_ham_dis_table = get_sorted_indices(l)
                    sorted_ham_all.append(sorted_ham_dis_table)

                is_used = [False] * len(sorted_feature_dis_indice)

                # 对排序后的特征距离从小到多逐个遍历，开始分配编码
                for idx in range(len(sorted_feature_dis_indice)):
                    sub_d_ind = sub_data_index_list[sorted_feature_dis_indice[idx]]  # 特征在源数据的索引

                    # 对某个父亲兄弟汉明距离表从小到多逐个遍历，选择没用过的
                    for idx2 in range(len(gen_code)):
                        the_code_idx = sorted_ham_all[min_bro[idx]][idx2]
                        if not is_used[the_code_idx]:  # 如果没被用过
                            is_used[the_code_idx] = True
                            self.bin_code_list[sub_d_ind] = [gen_code[the_code_idx], True]  # 分配
                            break


            # 如果子簇数量>10，则对每个中心最近的数据进行编码，对每个进一步进行kmeans
            else:
                centroid1, assment1 = bi_kMeans(data_filtered, k_next)  # 第一次聚类的k是指定的

                new_center_code = []  # 存储新的center编码，供下一轮递归使用
                new_center_index = []  # 存储新的center在源数据索引，供下一轮递归使用

                # 根据子聚类结果的中心，进行编码
                # 获得子聚类中心在源数据中的index
                min_distance_indices = find_min_distance_indices(assment1)
                sub_data_index_list = []    # 子簇内部中心在源数据的索引
                for center in min_distance_indices:
                    sub_data = data_filtered[center]
                    sub_data_index = np.where(np.all(self.data_mat == sub_data, axis=1))[0][0]
                    sub_data_index_list.append(sub_data_index)

        # ============step1：根据特征距离，为后面选择汉明距离做参考====================
                # 计算子聚类中心和各个父亲兄弟中心的特征距离
                feature_distance_table = generate_feature_distance_table(self.data_mat, sub_data_index_list, center_index, i)

                # 得到各个兄弟的特征距离后，取其中距离最近的元素，进行特征距离的排序，以指导汉明距离选择
                min_bro, min_feature_dis = find_min_bro(feature_distance_table)  # 记录是从哪个父亲兄弟获得的最小特征距离，以后分配码字考虑该父亲兄弟的汉明距离

                # 对最小特征距离表进行排序，返回排序后原特征距离的表的索引，用于指导汉明距离选择更近的
                sorted_feature_dis_indice = get_sorted_indices(min_feature_dis)

        # ============step2：根据特征距离，选择汉明距离编码====================
                # 根据父亲i进行翻转策略产生候选编码
                gen_code = find_unique_flipped_sample(self.bin_code_list, center_code[i], flip=flip, n=len(sorted_feature_dis_indice))

                # 计算子聚类中心候选编码和各个父亲兄弟中心的汉明距离
                ham_distance_table = generate_ham_distance_table(self.bin_code_list, gen_code, center_index, i)

        # ============step3：分配汉明编码====================
                # 对每个父亲兄弟中心的汉明距离表，返回排序后原特征距离的表的索引
                sorted_ham_all = []
                for l in ham_distance_table:
                    sorted_ham_dis_table = get_sorted_indices(l)
                    sorted_ham_all.append(sorted_ham_dis_table)

                is_used = [False] * len(sorted_feature_dis_indice)

                # 对排序后的特征距离从小到多逐个遍历，开始分配编码
                for idx in range(len(sorted_feature_dis_indice)):
                    sub_d_ind = sub_data_index_list[sorted_feature_dis_indice[idx]]     # 特征在源数据的索引

                    # 对某个父亲兄弟汉明距离表从小到多逐个遍历，选择没用过的
                    for idx2 in range(len(sorted_feature_dis_indice)):
                        the_code_idx = sorted_ham_all[min_bro[idx]][idx2]
                        if not is_used[the_code_idx]:   # 如果没被用过
                            is_used[the_code_idx] = True
                            self.bin_code_list[sub_d_ind] = [gen_code[the_code_idx], True]  # 分配
                            new_center_code.append(gen_code[the_code_idx])  # 添加为新中心的编码
                            new_center_index.append(sub_d_ind)
                            break

                # 编码完本簇的中心，仍需对子簇的子簇进行递归的编码，此时，子簇data_filtered作为父簇
                self.encode_func_sorted(new_center_code, new_center_index, assment1, data_filtered, self.code_len, 1)

    def encode_func_sorted_with_round(self, center_code, center_index, assment, data_father, k_next, round):
        """
        @param      center_code      上次聚类结果中心的编码列表，根据每一类父簇进行随机翻转
        @param      center_index     上次聚类结果中心对应向量在源数据中的index
        @param      assment     上次聚类结果，用来筛选父簇中某一子簇的数据，进行编码/进一步聚类+编码
        @param      data_father  父簇数据
        @param      k_next      下一层kmeans所需要分的类
        @param      flip        某轮的翻转策略，设计为第一轮C6_2=15，第二轮则为6
        """

        if round == 1:
            flip = 2
        elif round == 2:
            flip = 1
        else:
            flip = 1

        print("round = ", round)
        for i in range(len(center_code)):
            # 第i类子簇
            data_filtered = filter_data_by_cluster(assment, data_father, i)

            # 如果子簇数量<=10，停止聚类（否则空簇报错），对每个数据直接编码（size是为了方便判断，2是数据维度）
            if data_filtered.shape[0] <= self.code_len:
                sub_data_index_list = []  # 子簇内部中心在源数据的索引
                for j in range(data_filtered.shape[0]):
                    sub_data = data_filtered[j]
                    sub_data_index = np.where(np.all(self.data_mat == sub_data, axis=1))[0][0]
                    sub_data_index_list.append(sub_data_index)

                # ============step1：根据特征距离，为后面选择汉明距离做参考====================
                # 计算子聚类中心和各个父亲兄弟中心的特征距离
                feature_distance_table = generate_feature_distance_table(self.data_mat, sub_data_index_list, center_index, i)

                # 得到各个兄弟的特征距离后，取其中距离最近的元素，进行特征距离的排序，以指导汉明距离选择
                min_bro, min_feature_dis = find_min_bro(feature_distance_table)  # 记录是从哪个父亲兄弟获得的最小特征距离，以后分配码字考虑该父亲兄弟的汉明距离

                # 对最小特征距离表进行排序，返回排序后原特征距离的表的索引，用于指导汉明距离选择更近的
                sorted_feature_dis_indice = get_sorted_indices(min_feature_dis)

                # ============step2：根据特征距离，选择汉明距离编码====================
                # 根据父亲i进行翻转策略产生候选编码
                gen_code = find_unique_flipped_sample(self.bin_code_list, center_code[i], flip=flip, n=len(sorted_feature_dis_indice))

                # 计算子聚类中心候选编码和各个父亲兄弟中心的汉明距离
                ham_distance_table = generate_ham_distance_table(self.bin_code_list, gen_code, center_index, i)

                # ============step3：分配汉明编码====================
                # 对每个父亲兄弟中心的汉明距离表，返回排序后原特征距离的表的索引
                sorted_ham_all = []
                for l in ham_distance_table:
                    sorted_ham_dis_table = get_sorted_indices(l)
                    sorted_ham_all.append(sorted_ham_dis_table)

                is_used = [False] * len(sorted_feature_dis_indice)

                # 对排序后的特征距离从小到多逐个遍历，开始分配编码
                for idx in range(len(sorted_feature_dis_indice)):
                    sub_d_ind = sub_data_index_list[sorted_feature_dis_indice[idx]]  # 特征在源数据的索引

                    # 对某个父亲兄弟汉明距离表从小到多逐个遍历，选择没用过的
                    for idx2 in range(len(gen_code)):
                        the_code_idx = sorted_ham_all[min_bro[idx]][idx2]
                        if not is_used[the_code_idx]:  # 如果没被用过
                            is_used[the_code_idx] = True
                            self.bin_code_list[sub_d_ind] = [gen_code[the_code_idx], True]  # 分配
                            break


            # 如果子簇数量>10，则对每个中心最近的数据进行编码，对每个进一步进行kmeans
            else:
                centroid1, assment1 = bi_kMeans(data_filtered, k_next)  # 第一次聚类的k是指定的

                new_center_code = []  # 存储新的center编码，供下一轮递归使用
                new_center_index = []  # 存储新的center在源数据索引，供下一轮递归使用


                # 根据子聚类结果的元素，进行可视化标记
                # 获得子聚类每类元素在源数据中的index
                sub_class_indices = find_sub_class_indices(assment1)
                # 子聚类中每类在父数据中的index，我们要找源数据中的index
                count_label = 0
                for class_ in sub_class_indices:
                    for ele in class_:
                        sub_data = data_filtered[ele]
                        sub_data_index = np.where(np.all(self.data_mat == sub_data, axis=1))[0][0]
                        self.y[sub_data_index] = count_label
                    count_label = count_label + 1


                # 根据子聚类结果的中心，进行编码
                # 获得子聚类中心在源数据中的index
                min_distance_indices = find_min_distance_indices(assment1)
                sub_data_index_list = []    # 子簇内部中心在源数据的索引
                for center in min_distance_indices:
                    sub_data = data_filtered[center]
                    sub_data_index = np.where(np.all(self.data_mat == sub_data, axis=1))[0][0]
                    sub_data_index_list.append(sub_data_index)

                    self.y[sub_data_index] = self.y_center  # 可视化中心点
                self.y_center = self.y_center + 1

                # chart(self.data_mat, self.y)

        # ============step1：根据特征距离，为后面选择汉明距离做参考====================
                # 计算子聚类中心和各个父亲兄弟中心的特征距离
                feature_distance_table = generate_feature_distance_table(self.data_mat, sub_data_index_list, center_index, i)

                # 得到各个兄弟的特征距离后，取其中距离最近的元素，进行特征距离的排序，以指导汉明距离选择
                min_bro, min_feature_dis = find_min_bro(feature_distance_table)  # 记录是从哪个父亲兄弟获得的最小特征距离，以后分配码字考虑该父亲兄弟的汉明距离

                # 对最小特征距离表进行排序，返回排序后原特征距离的表的索引，用于指导汉明距离选择更近的
                sorted_feature_dis_indice = get_sorted_indices(min_feature_dis)

        # ============step2：根据特征距离，选择汉明距离编码====================
                # 根据父亲i进行翻转策略产生候选编码
                gen_code = find_unique_flipped_sample(self.bin_code_list, center_code[i], flip=flip, n=len(sorted_feature_dis_indice))

                # 计算子聚类中心候选编码和各个父亲兄弟中心的汉明距离
                ham_distance_table = generate_ham_distance_table(self.bin_code_list, gen_code, center_index, i)

        # ============step3：分配汉明编码====================
                # 对每个父亲兄弟中心的汉明距离表，返回排序后原特征距离的表的索引
                sorted_ham_all = []
                for l in ham_distance_table:
                    sorted_ham_dis_table = get_sorted_indices(l)
                    sorted_ham_all.append(sorted_ham_dis_table)

                is_used = [False] * len(sorted_feature_dis_indice)

                # 对排序后的特征距离从小到多逐个遍历，开始分配编码
                for idx in range(len(sorted_feature_dis_indice)):
                    sub_d_ind = sub_data_index_list[sorted_feature_dis_indice[idx]]     # 特征在源数据的索引

                    # 对某个父亲兄弟汉明距离表从小到多逐个遍历，选择没用过的
                    for idx2 in range(len(sorted_feature_dis_indice)):
                        the_code_idx = sorted_ham_all[min_bro[sorted_feature_dis_indice[idx]]][idx2]
                        if not is_used[the_code_idx]:   # 如果没被用过
                            is_used[the_code_idx] = True
                            self.bin_code_list[sub_d_ind] = [gen_code[the_code_idx], True]  # 分配
                            new_center_code.append(gen_code[the_code_idx])  # 添加为新中心的编码
                            new_center_index.append(sub_d_ind)
                            break

                # 编码完本簇的中心，仍需对子簇的子簇进行递归的编码，此时，子簇data_filtered作为父簇
                self.encode_func_sorted_with_round(new_center_code, new_center_index, assment1, data_filtered, self.code_len, round=round+1)


    def generate_decimal_index(self):
        # 第一次k=2聚类，将两个中心点的编码为'000000'，'111111'
        centroid, cluster_assment = bi_kMeans(self.data_mat, 2)
        # plot_cluster(data_mat, cluster_assment, centroid)

        center_code = ['0000000000', '1111111111']  # 中心的编码
        center_index = find_min_distance_indices(cluster_assment)

        self.bin_code_list[center_index[0]] = ['0000000000', True]
        self.bin_code_list[center_index[1]] = ['1111111111', True]

        self.y = np.array(cluster_assment[:, 0].reshape(len(cluster_assment)))[0]

        for center in center_index:
            self.y[center] = self.y_center
        self.y_center = self.y_center + 1

        chart(codebook_reduce, self.y)

        # self.encode_func_random(center_code, cluster_assment, self.data_mat, k_next=15, flip=3)  # 第一种，不需要center_index参数，第一轮flip=3，后面的全部flip=1递增
        # self.encode_func_random_with_round(center_code, cluster_assment, self.data_mat, k_next=15, round=1)  # 第一种轮数改进，规定第一二轮flip
        # self.encode_func_random1(center_code, cluster_assment, self.data_mat, k_next=15, flip=3)  # 第一种，不需要center_index参数，第一轮flip=3，后面的全部flip=1递增
        # self.encode_func_random1_with_round(center_code, cluster_assment, self.data_mat, k_next=16, round=1)  # 第一种轮数改进，规定第一二轮flip
        # self.encode_func_sorted(center_code, center_index, cluster_assment, self.data_mat, k_next=20, flip=4)    # 第二种，需要center_index参数
        self.encode_func_sorted_with_round(center_code, center_index, cluster_assment, self.data_mat, k_next=15, round=1)  # 第二种轮数改进，需要center_index参数，规定第一二轮flip


        finish_coding = check_duplicates(self.bin_code_list)
        print(finish_coding)

        decimal_index = bin_code_to_decimal_index(self.bin_code_list)
        print(decimal_index)

        missing_num = find_missing_numbers(decimal_index)
        dup_index = find_duplicate_indices(decimal_index)

        for i in range(len(missing_num)):
            decimal_index[dup_index[i]] = missing_num[i]

        print('完成覆盖')

        missing_num = find_missing_numbers(decimal_index)
        dup_index = find_duplicate_indices(decimal_index)

        print(decimal_index)
        return decimal_index


def chart(X, y):
    # --------------------------------------------------------------------------#
    # This section is not mandatory as its purpose is to sort the data by label
    # so, we can maintain consistent colors for digits across multiple graphs

    # Concatenate X and y arrays
    arr_concat = np.concatenate((X, y.reshape(y.shape[0], 1)), axis=1)
    # Create a Pandas dataframe using the above array
    df = pd.DataFrame(arr_concat, columns=['x', 'y', 'z', 'label'])
    # Convert label data type from float to integer
    df['label'] = df['label'].astype(int)
    # Finally, sort the dataframe by label
    df.sort_values(by='label', axis=0, ascending=True, inplace=True)
    # --------------------------------------------------------------------------#
    # Create a 3D graph
    fig = px.scatter_3d(df, x='x', y='y', z='z', color=df['label'].astype(str), height=900, width=950)

    # Update chart looks
    fig.update_layout(title_text='UMAP',
                      showlegend=True,
                      legend=dict(orientation="h", yanchor="top", y=0, xanchor="center", x=0.5),
                      scene_camera=dict(up=dict(x=0, y=0, z=1),
                                        center=dict(x=0, y=0, z=-0.1),
                                        eye=dict(x=1.5, y=-1.4, z=0.5)),
                      margin=dict(l=0, r=0, b=0, t=0),
                      scene=dict(xaxis=dict(backgroundcolor='white',
                                            color='black',
                                            gridcolor='#f0f0f0',
                                            title_font=dict(size=10),
                                            tickfont=dict(size=10),
                                            ),
                                 yaxis=dict(backgroundcolor='white',
                                            color='black',
                                            gridcolor='#f0f0f0',
                                            title_font=dict(size=10),
                                            tickfont=dict(size=10),
                                            ),
                                 zaxis=dict(backgroundcolor='lightgrey',
                                            color='black',
                                            gridcolor='#f0f0f0',
                                            title_font=dict(size=10),
                                            tickfont=dict(size=10),
                                            )))
    # Update marker size
    fig.update_traces(marker=dict(size=3, line=dict(color='black', width=0.1)))

    fig.show()


def umap_reduce(X):
    reducer = umap.UMAP(n_neighbors=15,
                        # default 15, The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation.
                        n_components=3,  # default 2, The dimension of the space to embed into.
                        metric='euclidean',
                        # default 'euclidean', The metric to use to compute distances in high dimensional space.
                        n_epochs=1000,
                        # default None, The number of training epochs to be used in optimizing the low dimensional embedding. Larger values result in more accurate embeddings.
                        learning_rate=1.0,  # default 1.0, The initial learning rate for the embedding optimization.
                        init='spectral',
                        # default 'spectral', How to initialize the low dimensional embedding. Options are: {'spectral', 'random', A numpy array of initial embedding positions}.
                        min_dist=0.1,  # default 0.1, The effective minimum distance between embedded points.
                        spread=1.0,
                        # default 1.0, The effective scale of embedded points. In combination with ``min_dist`` this determines how clustered/clumped the embedded points are.
                        low_memory=False,
                        # default False, For some datasets the nearest neighbor computation can consume a lot of memory. If you find that UMAP is failing due to memory constraints consider setting this option to True.
                        set_op_mix_ratio=1.0,
                        # default 1.0, The value of this parameter should be between 0.0 and 1.0; a value of 1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy intersection.
                        local_connectivity=1,
                        # default 1, The local connectivity required -- i.e. the number of nearest neighbors that should be assumed to be connected at a local level.
                        repulsion_strength=1.0,
                        # default 1.0, Weighting applied to negative samples in low dimensional embedding optimization.
                        negative_sample_rate=5,
                        # default 5, Increasing this value will result in greater repulsive force being applied, greater optimization cost, but slightly more accuracy.
                        transform_queue_size=4.0,
                        # default 4.0, Larger values will result in slower performance but more accurate nearest neighbor evaluation.
                        a=None,
                        # default None, More specific parameters controlling the embedding. If None these values are set automatically as determined by ``min_dist`` and ``spread``.
                        b=None,
                        # default None, More specific parameters controlling the embedding. If None these values are set automatically as determined by ``min_dist`` and ``spread``.
                        random_state=42,
                        # default: None, If int, random_state is the seed used by the random number generator;
                        metric_kwds=None,
                        # default None) Arguments to pass on to the metric, such as the ``p`` value for Minkowski distance.
                        angular_rp_forest=False,
                        # default False, Whether to use an angular random projection forest to initialise the approximate nearest neighbor search.
                        target_n_neighbors=-1,
                        # default -1, The number of nearest neighbors to use to construct the target simplcial set. If set to -1 use the ``n_neighbors`` value.
                        # target_metric='categorical', # default 'categorical', The metric used to measure distance for a target array is using supervised dimension reduction. By default this is 'categorical' which will measure distance in terms of whether categories match or are different.
                        # target_metric_kwds=None, # dict, default None, Keyword argument to pass to the target metric when performing supervised dimension reduction. If None then no arguments are passed on.
                        # target_weight=0.5, # default 0.5, weighting factor between data topology and target topology.
                        transform_seed=42,
                        # default 42, Random seed used for the stochastic aspects of the transform operation.
                        verbose=False,  # default False, Controls verbosity of logging.
                        unique=False,
                        # default False, Controls if the rows of your data should be uniqued before being embedded.
                        )
    # Fit and transform the data
    X_trans = reducer.fit_transform(X)

    # Check the shape of the new data
    print('Shape of X_trans: ', X_trans.shape)

    X_final = torch.from_numpy(X_trans)

    return X_final