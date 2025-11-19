from itertools import combinations
import codecs
from numpy import *
import matplotlib.pyplot as plt
import torch
import numpy as np
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.utils.misc import get_device

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


def load_data(path):
    """
    @brief      Loads a data.
    @param      path  The path
    @return     data set
    """
    data_set = list()
    with codecs.open(path) as f:
        for line in f.readlines():
            data = line.strip().split("\t")
            flt_data = list(map(float, data))
            data_set.append(flt_data)
    return data_set


def rand_cent(data_mat, k):
    """
    @brief      select random centroid
    @param      data_mat  The data matrix
    @param      k
    @return     centroids
    """
    n = shape(data_mat)[1]
    centroids = mat(zeros((k, n)))
    if not data_mat.any():
        return centroids
    for j in range(n):
        minJ = min(data_mat[:, j])
        rangeJ = float(max(data_mat[:, j]) - minJ)
        centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))
    return centroids


def dist_eucl(vecA, vecB):
    """
    @brief      the similarity function
    @param      vecA  The vector a
    @param      vecB  The vector b
    @return     the euclidean distance
    """
    return sqrt(sum(power(vecA - vecB, 2)))


def get_closest_dist(point, centroid):
    """
    @brief      Gets the closest distance.
    @param      point     The point
    @param      centroid  The centroid
    @return     The closest distance.
    """
    # 计算与已有质心最近的距离
    min_dist = inf
    for j in range(len(centroid)):
        distance = dist_eucl(point, centroid[j])
        if distance < min_dist:
            min_dist = distance
    return min_dist


def kpp_cent(data_mat, k):
    """
    @brief      kmeans++ init centor
    @param      data_mat  The data matrix
    @param      k   num of cluster
    @return     init centroid
    """
    data_set = data_mat.getA()
    # 随机初始化第一个中心点
    centroid = list()
    centroid.append(data_set[random.randint(0, len(data_set))])
    d = [0 for i in range(len(data_set))]
    for _ in range(1, k):
        total = 0.0
        for i in range(len(data_set)):
            d[i] = get_closest_dist(data_set[i], centroid)
            total += d[i]
        total *= random.rand()
        # 选取下一个中心点
        for j in range(len(d)):
            total -= d[j]
            if total > 0:
                continue
            centroid.append(data_set[j])
            break
    return mat(centroid)


def k_Means(data_mat, k, dist="dist_eucl", create_cent="kpp_cent"):
    """
    @brief      kMeans algorithm
    @param      data_mat     The data matrix
    @param      k            num of cluster
    @param      dist         The distance funtion
    @param      create_cent  The create centroid function
    @return     the cluster
    """
    m = shape(data_mat)[0]
    # 初始化点的簇
    cluster_assment = mat(zeros((m, 2)))  # 类别，距离
    # 随机初始化聚类初始点
    centroid = eval(create_cent)(data_mat, k)
    cluster_changed = True
    # 遍历每个点
    while cluster_changed:
        cluster_changed = False
        for i in range(m):
            min_index = -1
            min_dist = inf
            for j in range(k):
                distance = eval(dist)(data_mat[i, :], centroid[j, :])
                if distance < min_dist:
                    min_dist = distance
                    min_index = j
            if cluster_assment[i, 0] != min_index:
                cluster_changed = True
            cluster_assment[i, :] = min_index, min_dist ** 2
        # 计算簇中所有点的均值并重新将均值作为质心
        for j in range(k):
            per_data_set = data_mat[nonzero(cluster_assment[:, 0].A == j)[0]]
            centroid[j, :] = mean(per_data_set, axis=0)
    return centroid, cluster_assment


def bi_kMeans(data_mat, k, dist="dist_eucl"):
    """
    @brief      kMeans algorithm
    @param      data_mat     The data matrix
    @param      k            num of cluster
    @param      dist         The distance funtion
    @return     the cluster
    """
    m = shape(data_mat)[0]

    # 初始化点的簇
    cluster_assment = mat(zeros((m, 2)))  # 类别，距离

    # 初始化聚类初始点
    centroid0 = mean(data_mat, axis=0).tolist()[0]
    cent_list = [centroid0]
    # print(cent_list)

    # 初始化SSE
    for j in range(m):
        cluster_assment[j, 1] = eval(dist)(mat(centroid0), data_mat[j, :]) ** 2

    # 如果簇中心数目小于k，则继续在每个簇中进行k=2划分
    round = 0
    while (len(cent_list) < k):
        lowest_sse = inf
        # 继续在每类簇中进行k=2划分
        for i in range(len(cent_list)):
            # 尝试在每一类簇中进行k=2的kmeans划分
            ptsin_cur_cluster = data_mat[nonzero(cluster_assment[:, 0].A == i)[0], :]

            centroid_mat, split_cluster_ass = k_Means(ptsin_cur_cluster, k=2)

            # 计算分类之后的SSE值
            sse_split = sum(split_cluster_ass[:, 1])
            sse_nonsplit = sum(cluster_assment[nonzero(cluster_assment[:, 0].A != i)[0], 1])
            # print("sse_split, sse_nonsplit", sse_split, sse_nonsplit)
            # 记录最好的划分位置
            if sse_split + sse_nonsplit < lowest_sse:
                best_cent_tosplit = i
                best_new_cents = centroid_mat
                best_cluster_ass = split_cluster_ass.copy()
                lowest_sse = sse_split + sse_nonsplit

        # print('the bestCentToSplit is: ', best_cent_tosplit)
        # print('the len of bestClustAss is: ', len(best_cluster_ass))
        # 更新簇的分配结果
        best_cluster_ass[nonzero(best_cluster_ass[:, 0].A == 1)[0], 0] = len(cent_list)
        best_cluster_ass[nonzero(best_cluster_ass[:, 0].A == 0)[0], 0] = best_cent_tosplit
        cent_list[best_cent_tosplit] = best_new_cents[0, :].tolist()[0]
        cent_list.append(best_new_cents[1, :].tolist()[0])
        cluster_assment[nonzero(cluster_assment[:, 0].A == best_cent_tosplit)[0], :] = best_cluster_ass

        # 每一次产生进一步划分后，作图
        round += 1
        # plot_cluster(data_mat, cluster_assment, mat(cent_list))
    return mat(cent_list), cluster_assment


def plot_cluster(data_mat, cluster_assment, centroid):
    """
    @brief      plot cluster and centroid
    @param      data_mat        The data matrix
    @param      cluster_assment  The cluste assment
    @param      centroid        The centroid
    @return
    """
    plt.figure(figsize=(15, 6), dpi=80)
    plt.subplot(121)
    plt.plot(data_mat[:, 0], data_mat[:, 1], 'o')

    plt.xlim(-5, 5)
    plt.ylim(-5, 5.5)

    plt.title("source data", fontsize=15)
    plt.subplot(122)
    k = shape(centroid)[0]
    colors = [plt.cm.Spectral(each) for each in linspace(0, 1, k)]
    for i, col in zip(range(k), colors):
        per_data_set = data_mat[nonzero(cluster_assment[:, 0].A == i)[0]]
        plt.plot(per_data_set[:, 0], per_data_set[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=10)
    for i in range(k):
        plt.plot(centroid[:, 0], centroid[:, 1], '+', color='k', markersize=18)

    plt.xlim(-5, 5)
    plt.ylim(-5, 5.5)
    plt.title("bi_KMeans Cluster, k = 3", fontsize=15)
    plt.show()


def filter_data_by_cluster(cluster_assment, data_mat, index):
    # 筛选数据中指定类别的簇
    data_mat = np.array(data_mat)
    cluster_assment = np.array(cluster_assment)
    mask = cluster_assment[:, 0] == index
    filtered_data = data_mat[mask]
    data_mat1 = np.copy(filtered_data)

    return mat(data_mat1)


# 将sample随机翻转flip位码字，生成码字不和总表bin中已使用的码字重复
def generate_flipped_samples(sample, flip):
    n = len(sample)
    flip_indices_combinations = combinations(range(n), flip)  # 生成所有可能的翻转位索引组合
    flipped_samples = []
    for indices in flip_indices_combinations:
        flipped_sample = ''.join(
            ['1' if i in indices and sample[i] == '0' else '0' if i in indices and sample[i] == '1' else sample[i] for i
             in range(n)])
        flipped_samples.append(flipped_sample)
    return flipped_samples

# 将sample翻转flip位，不与bin中使用过的重复，共生成n个翻转后结果
def find_unique_flipped_sample(bin, sample, flip, n):
    n_flipped_sample = []
    flip_list = []
    while True:
        flipped_samples = generate_flipped_samples(sample, flip)
        # random.shuffle(flipped_samples)
        for flipped_sample in flipped_samples:
            conflicting = False
            for item in bin:
                if item[1] and item[0] == flipped_sample:
                    conflicting = True
                    break
            if not conflicting:
                n_flipped_sample.append(flipped_sample)
                flip_list.append(flip)
            if len(n_flipped_sample) == n:
                # print(flip_list)
                return n_flipped_sample
        flip += 1
        if flip >= 10:
            return n_flipped_sample


# 每次聚类后，将其类别可视化
def find_sub_class_indices(assment):
    assment = np.array(assment)
    # 获取聚类类别值
    categories = np.unique(assment[:, 0])
    indices = [[] for _ in range(len(categories))]  # 创建一个包含k类别个空列表的列表

    # 遍历每个元素，查看属于哪一个类，索引添加到该类的列表中
    for i in range(len(assment)):
        category = int(assment[i, 0])  # 获取二元组的类别
        indices[category].append(i)  # 将索引添加到对应类别的列表中

    return indices


# 输入聚类结果assment，返回距离聚类中心最近的索引，用于将中心编码
def find_min_distance_indices(assment):
    assment = np.array(assment)
    # 获取聚类类别值
    categories = np.unique(assment[:, 0])
    min_distance_indices = []

    # 遍历每个类别
    for category in categories:
        # 从数组中筛选出当前类别的子集
        category_subset = assment[assment[:, 0] == category]
        # 找到当前类别中距离最小的索引
        min_distance_index = np.argmin(category_subset[:, 1])
        # 找到当前类别中距离最小的数组在原始data数组中的索引
        original_index = np.where(assment[:, 0] == category)[0][min_distance_index]
        # 将最小索引添加到列表中
        min_distance_indices.append(original_index)

    return min_distance_indices


# def encode_func(father, assment, data_father, k_next, flip):
#     """
#     @param      father      上次聚类结果中心的编码列表，根据每一类父簇进行随机翻转
#     @param      assment     上次聚类结果，用来筛选父簇中某一子簇的数据，进行编码/进一步聚类+编码
#     @param      data_father  父簇数据
#     @param      k_next      下一层kmeans所需要分的类
#     @param      flip        某轮的翻转策略，设计为第一轮C6_2=15，第二轮则为6
#     """
#     for i in range(len(father)):
#         # 子簇
#         data_filtered = filter_data_by_cluster(assment, data_father, i)
#
#         # 如果子簇数量<=6，停止聚类（否则空簇报错），对每个数据直接编码（size是为了方便判断，2是数据维度）
#         if data_filtered.shape[0] <= code_len:
#             for j in range(data_filtered.shape[0]):
#                 sub_data = data_filtered[j]
#                 sub_data_index = np.where(np.all(data_mat == sub_data, axis=1))[0][0]
#
#                 # 如果未进行编码，则调用随机翻转算法进行编码
#                 if not bin_code_list[sub_data_index][1]:
#                     gen_code = find_unique_flipped_sample(bin_code_list, father[i], flip=flip)
#                     bin_code_list[sub_data_index] = [gen_code, True]
#                 # 否则就不用处理，直接进行下一个sub_data的处理
#
#
#         # 如果子簇数量>6，则对每个中心最近的数据进行编码，对每个进一步进行kmeans
#         else:
#             centroid1, assment1 = bi_kMeans(data_filtered, k_next)  # 第一次聚类的k是指定的
#
#             new_father = []  # 存储新的father编码，供下一轮递归使用
#             # 根据子聚类结果的中心，进行编码
#             min_distance_indices = find_min_distance_indices(assment1)
#             for center in min_distance_indices:
#                 sub_data = data_filtered[center]
#                 sub_data_index = np.where(np.all(data_mat == sub_data, axis=1))[0][0]
#
#                 # 如果未进行编码，则调用随机翻转算法进行编码
#                 if not bin_code_list[sub_data_index][1]:
#                     gen_code = find_unique_flipped_sample(bin_code_list, father[i], flip=flip)
#                     bin_code_list[sub_data_index] = [gen_code, True]
#                     new_father.append(gen_code)
#                 # 否则就不用处理，直接进行下一个sub_data的处理
#
#             # 编码完本簇的中心，仍需对子簇的子簇进行递归的编码，此时，子簇data_filtered作为父簇
#             encode_func(new_father, assment1, data_filtered, code_len, 1)

def check_duplicates(bin):
    string_set = set()
    for item in bin:
        string = item[0]
        if string in string_set:
            return True
        string_set.add(string)
    return False


def bin_code_to_decimal_index(bin_list):
    decimal_list = []
    for item in bin_list:
        binary_string = item[0]
        decimal_number = int(binary_string, 2)
        decimal_list.append(decimal_number)
    return decimal_list


# 找到重复的index列表，给他们直接分配0-1023未使用的索引
def find_duplicate_indices(lst):
    duplicate_indices = {}
    for i, num in enumerate(lst):
        if num in duplicate_indices:
            duplicate_indices[num].append(i)
        else:
            duplicate_indices[num] = [i]
    result = [indices[1:] for indices in duplicate_indices.values() if len(indices) > 1]
    merged_indices = [index for indices in result for index in indices]
    return merged_indices


# 0-1023未使用的索引
def find_missing_numbers(lst):
    all_numbers = set(range(1024))
    present_numbers = set(lst)
    missing_numbers = list(all_numbers - present_numbers)
    return missing_numbers


# 计算子数据与所有父亲兄弟的特征距离，data是源数据，sub、center是代表子数据和中心数据的索引，father用于排除计算父亲，只计算父亲兄弟
def generate_feature_distance_table(data, sub, center, father):
    feature_distance_table = []

    count = 0
    for c in center:
        # 排除计算father的距离，只计算和父亲兄弟的距离
        if father != count:
            distances = []
            center_element = data[c]

            for s in sub:
                sub_element = data[s]
                distance = np.linalg.norm(center_element - sub_element)
                distances.append(distance)

            feature_distance_table.append(distances)
        count += 1
    return feature_distance_table


# 计算子数据与所有父亲兄弟的汉明距离，bin_data是源数据的编码表，candidate_bin_code是候选编码，center是中心数据索引，father用于排除计算父亲，只计算父亲兄弟
def generate_ham_distance_table(bin_data, candidate_bin_code, center, father):
    ham_distance_table = []

    count = 0
    for c in center:
        # 排除计算father的距离，只计算和父亲兄弟的距离
        if father != count:
            distances = []
            center_bin_code = bin_data[c][0]

            for bin in candidate_bin_code:
                distance = ham_distance(bin, center_bin_code)
                distances.append(distance)

            ham_distance_table.append(distances)
        count += 1
    return ham_distance_table


def ham_distance(str1, str2):
    if len(str1) != len(str2):
        raise ValueError("字符串长度不一致")

    distance = 0
    for i in range(len(str1)):
        if str1[i] != str2[i]:
            distance += 1

    return distance

# 找多个父亲兄弟间，距离最近的索引表；表内容代表这个值是跟哪个父亲兄弟特征距离最近，用于参考哪个父亲兄弟的码字距离
def find_min_bro(lists):
    k = len(lists[0])
    n = len(lists)

    result_index = []
    result = []
    for i in range(k):
        min_index = 0
        min_value = lists[0][i]

        for j in range(1, n):
            if lists[j][i] < min_value:
                min_index = j
                min_value = lists[j][i]

        result_index.append(min_index)
        result.append(min_value)

    return result_index, result

# 找多个父亲兄弟间，距离最近的索引表；表内容代表这个值是跟哪个父亲兄弟特征距离最近，用于参考哪个父亲兄弟的码字距离
def find_min_bro(lists):
    k = len(lists[0])
    n = len(lists)

    result_index = []
    result = []
    for i in range(k):
        min_index = 0
        min_value = lists[0][i]

        for j in range(1, n):
            if lists[j][i] < min_value:
                min_index = j
                min_value = lists[j][i]

        result_index.append(min_index)
        result.append(min_value)

    return result_index, result


# 对列表从小到大排序，返回排完序后的源数据索引表
def get_sorted_indices(lst):
    indices = sorted(range(len(lst)), key=lambda x: lst[x])
    return indices


"""
    全局变量
"""
# 数据数量64，要给64条数据进行6比特的编码分配
# data_mat = mat(load_data("data/testSet2_kmeans.txt"))
data_mat = mat(vqgan.quantize.embedding.weight.tolist())

# 编码存储数组，初始化为全0，确定编码后第二个值为True
bin_code_list = [['0000000000', False]] * data_mat.shape[0]

#  数据维度
data_dimension = 256

# 码字长度
code_len = 10