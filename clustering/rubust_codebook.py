from clustering.bi_kmeans import *
import utils
from clustering.umap_reduce import umap_reduce

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


class Rubust_Codebook(object):
    """
    @brief  把鲁棒性codebook写成类，方便使用
    @args   codebook_data:      外部指定的VQGAN的codebook参数
    @args   dimension:          数据维度：256
    @args   code_len:           需要编码的码字长度，如数据长度为1024，则码字长度为10 (1024 = 2^10)
    """

    def __init__(self, codebook_data, code_len):
        self.data_mat = mat(codebook_data.tolist())  # 替代data_mat = mat(vqgan.quantize.embedding.weight.tolist())
        self.bin_code_list = [['0000000000', False]] * self.data_mat.shape[0]  # 编码存储数组，初始化为全0，确定编码后第二个值为True
        # self.data_dimension = dimension  # 数据维度
        self.code_len = code_len  # 码字长度

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
        @param      k_next      下一层kmeans所需要分的类, 设计为第一轮C6_2=15，第二轮则为C6_1=6
        @param      flip        某轮的翻转策略，max(n/4, 1)
        """
        if round == 1:
            # flip = 4
            flip = 8
        elif round == 2:
            flip = 2
        else:
            flip = 1

        # print("round = ", round)

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
                # k_next = math.comb(self.code_len, flip) # 计算下一轮的k值
                self.encode_func_random1_with_round(new_center_code, assment1, data_filtered, self.code_len, round=round + 1)
                # self.encode_func_random1_with_round(new_center_code, assment1, data_filtered, k_next, round=round + 1)

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
        @param      k_next      下一层kmeans所需要分的类，设计为第一轮C10_2=45，第二轮则为 C10_1 = 10
        @param      flip        某轮的翻转策略，设计为第一轮8，第二轮则为 Max(8/4,1)
        """

        if round == 1:
            flip = 2
            # flip = 8
        elif round == 2:
            flip = 1
            # flip = 2
        else:
            flip = 1

        # print("round = ", round)
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
                    for idx2 in range(len(sorted_feature_dis_indice)):
                        the_code_idx = sorted_ham_all[min_bro[sorted_feature_dis_indice[idx]]][idx2]
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

        # self.encode_func_random(center_code, cluster_assment, self.data_mat, k_next=15, flip=3)  # 第一种，不需要center_index参数，第一轮flip=3，后面的全部flip=1递增
        # self.encode_func_random_with_round(center_code, cluster_assment, self.data_mat, k_next=15, round=1)  # 第一种轮数改进，规定第一二轮flip
        # self.encode_func_random1(center_code, cluster_assment, self.data_mat, k_next=15, flip=3)  # 第一种，不需要center_index参数，第一轮flip=3，后面的全部flip=1递增
        self.encode_func_random1_with_round(center_code, cluster_assment, self.data_mat, k_next=20, round=1)  # 第一种轮数改进，规定第一二轮flip
        # self.encode_func_sorted(center_code, center_index, cluster_assment, self.data_mat, k_next=20, flip=4)    # 第二种，需要center_index参数
        # self.encode_func_sorted_with_round(center_code, center_index, cluster_assment, self.data_mat, k_next=15, round=1)  # 第二种轮数改进，需要center_index参数，规定第一二轮flip


        finish_coding = check_duplicates(self.bin_code_list)
        # print(finish_coding)
        decimal_index = bin_code_to_decimal_index(self.bin_code_list)
        # print(decimal_index)

        missing_num = find_missing_numbers(decimal_index)
        dup_index = find_duplicate_indices(decimal_index)

        for i in range(len(missing_num)):
            decimal_index[dup_index[i]] = missing_num[i]

        # print('完成覆盖')
        #
        # missing_num = find_missing_numbers(decimal_index)
        # dup_index = find_duplicate_indices(decimal_index)
        #
        # print(decimal_index)
        return decimal_index