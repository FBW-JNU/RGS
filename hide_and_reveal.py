import cv2
import argparse
import glob
import os
import pickle
import skimage
from torchvision.transforms.functional import normalize
from edict_functions import *
import numpy as np
import utils
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.misc import gpu_is_available, get_device
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.metrics import calculate_psnr, calculate_ssim


if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = get_device()
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path', type=str, default='./inputs/',
                        help='Input image, video or folder.')
    parser.add_argument('-t', '--stego_path', type=str, default='./stegos/',
                        help='Input image, video or folder.')
    parser.add_argument('-w', '--fidelity_weight', type=float, default=0.5,
                        help='Balance the quality and fidelity.')

    args = parser.parse_args()

    # ------------------ set up CodeFormer restorer -------------------
    vqgan = ARCH_REGISTRY.get('VQAutoEncoder')(img_size=512, nf=64, ch_mult=[1, 2, 2, 4, 4, 8],
                                               quantizer='nearest', res_blocks=2, attn_resolutions=[16],
                                               codebook_size=1024).to(device)

    # VQGAN模型权重
    ckpt_path = './weights/vqgan_code1024.pth'
    checkpoint = torch.load(ckpt_path)['params_ema']
    vqgan.load_state_dict(checkpoint)
    vqgan.eval()

    # ========  使用预设固定鲁棒codebook（不执行鲁棒聚类算法）
    codebook = np.array(vqgan.quantize.embedding.weight.tolist())
    codebook = torch.Tensor(codebook)
    
    # 若：执行鲁棒性codebook算法
    # codebook_reduce = umap_reduce(codebook)
    #
    # rubust = Rubust_Codebook(codebook_reduce, code_len=10)
    # rubust_decimal_index = rubust.generate_decimal_index()
    
    # 若：直接使用鲁棒性算法生成的固定codebook index
    rubust_decimal_index = [685, 239, 188, 203, 122, 540, 797, 458, 790, 1003, 42, 892, 151, 368, 119, 272, 864, 31,
                            948, 300, 278, 1007, 319, 44, 830, 350, 833, 619, 962, 93, 312, 591, 871, 692, 399, 899,
                            1015, 897, 1022, 911, 301, 265, 1021, 295, 835, 519, 770, 913, 61, 110, 322, 247, 76, 490,
                            347, 127, 208, 878, 537, 141, 920, 254, 822, 30, 109, 607, 60, 629, 432, 57, 304, 914, 94,
                            115, 844, 248, 584, 734, 749, 677, 514, 576, 1020, 627, 750, 994, 640, 856, 820, 966, 663,
                            527, 171, 80, 802, 289, 918, 91, 756, 735, 708, 205, 507, 894, 223, 62, 850, 418, 125, 639,
                            691, 845, 167, 451, 496, 681, 902, 219, 516, 532, 245, 553, 666, 939, 903, 92, 572, 149, 67,
                            211, 652, 51, 159, 866, 124, 273, 818, 646, 529, 455, 995, 838, 566, 759, 968, 924, 184,
                            554, 707, 933, 535, 107, 928, 937, 52, 901, 58, 360, 235, 199, 637, 329, 420, 930, 431, 650,
                            117, 588, 841, 49, 146, 271, 761, 178, 615, 97, 852, 649, 81, 517, 213, 303, 689, 390, 372,
                            281, 964, 929, 506, 19, 16, 480, 625, 243, 74, 525, 526, 559, 720, 305, 111, 862, 849, 774,
                            952, 916, 192, 960, 505, 100, 883, 599, 760, 737, 684, 857, 262, 870, 842, 875, 186, 153,
                            779, 977, 165, 285, 84, 620, 446, 926, 121, 237, 308, 172, 118, 38, 944, 126, 241, 611, 782,
                            728, 851, 590, 452, 259, 260, 233, 751, 616, 520, 884, 470, 50, 578, 642, 491, 1008, 992,
                            796, 983, 927, 828, 408, 234, 104, 4, 497, 600, 437, 644, 468, 922, 483, 357, 425, 617, 282,
                            558, 793, 155, 861, 932, 935, 557, 500, 160, 869, 837, 740, 323, 768, 829, 732, 675, 73,
                            648, 318, 976, 215, 502, 743, 636, 881, 785, 32, 539, 348, 185, 865, 736, 384, 229, 409,
                            873, 794, 568, 972, 196, 284, 473, 240, 457, 351, 769, 817, 261, 191, 476, 0, 474, 253, 594,
                            378, 11, 274, 478, 826, 868, 546, 337, 848, 716, 154, 26, 143, 672, 618, 258, 579, 15, 136,
                            533, 953, 550, 524, 936, 70, 511, 157, 695, 1019, 673, 877, 879, 767, 821, 394, 87, 69, 294,
                            72, 747, 628, 477, 13, 335, 68, 961, 887, 2, 413, 569, 453, 407, 513, 279, 908, 941, 1017,
                            250, 731, 585, 426, 417, 298, 331, 112, 1001, 565, 64, 641, 179, 608, 158, 589, 421, 1000,
                            766, 346, 923, 839, 168, 493, 658, 246, 129, 423, 781, 66, 415, 840, 90, 88, 680, 194, 609,
                            169, 33, 706, 98, 427, 662, 721, 819, 139, 443, 333, 363, 988, 562, 898, 564, 965, 601, 123,
                            344, 931, 836, 597, 997, 659, 545, 778, 984, 773, 660, 1012, 575, 297, 667, 556, 816, 938,
                            23, 1011, 580, 314, 495, 669, 613, 940, 745, 101, 296, 325, 664, 77, 47, 471, 321, 963, 249,
                            711, 655, 541, 330, 200, 583, 813, 164, 444, 75, 786, 447, 270, 96, 742, 893, 979, 998, 481,
                            256, 859, 738, 419, 128, 402, 14, 624, 696, 614, 224, 582, 231, 741, 59, 228, 206, 365, 606,
                            448, 449, 621, 364, 753, 907, 232, 632, 499, 396, 214, 990, 182, 197, 530, 317, 22, 832,
                            959, 440, 792, 108, 173, 367, 48, 509, 406, 190, 217, 183, 989, 375, 700, 316, 361, 193,
                            757, 800, 238, 494, 489, 645, 324, 366, 53, 403, 888, 633, 690, 181, 812, 465, 595, 230,
                            436, 982, 518, 7, 891, 302, 622, 113, 105, 131, 120, 152, 174, 99, 567, 863, 342, 462, 268,
                            880, 275, 958, 145, 912, 772, 612, 463, 503, 725, 671, 221, 9, 587, 957, 78, 405, 1023, 280,
                            411, 776, 512, 355, 434, 727, 699, 705, 764, 693, 161, 698, 570, 380, 999, 25, 132, 135,
                            510, 142, 971, 683, 987, 860, 654, 846, 544, 670, 218, 36, 810, 784, 225, 400, 986, 795,
                            210, 313, 209, 676, 43, 755, 748, 723, 492, 969, 156, 886, 909, 956, 758, 386, 8, 715, 548,
                            397, 293, 814, 1006, 638, 895, 266, 349, 310, 555, 523, 488, 454, 605, 12, 905, 162, 744,
                            610, 1010, 854, 925, 563, 220, 665, 765, 730, 45, 336, 985, 147, 201, 352, 82, 508, 1014,
                            889, 353, 634, 466, 910, 996, 264, 216, 341, 472, 370, 315, 328, 252, 307, 809, 528, 967,
                            682, 674, 626, 954, 993, 586, 116, 522, 872, 834, 631, 202, 263, 343, 299, 719, 799, 746,
                            441, 647, 574, 501, 338, 788, 383, 970, 577, 362, 603, 798, 339, 356, 257, 276, 805, 459,
                            137, 148, 29, 950, 140, 710, 701, 1018, 643, 808, 688, 867, 376, 551, 754, 1004, 10, 661,
                            166, 549, 885, 189, 354, 917, 981, 340, 780, 890, 251, 311, 560, 915, 653, 722, 28, 831,
                            373, 515, 464, 71, 24, 771, 783, 763, 460, 974, 900, 991, 40, 416, 371, 604, 41, 504, 733,
                            445, 461, 17, 242, 980, 212, 538, 277, 775, 630, 377, 714, 46, 236, 561, 291, 712, 227, 823,
                            55, 287, 222, 717, 180, 381, 843, 20, 686, 542, 921, 485, 410, 288, 54, 320, 801, 5, 1009,
                            651, 571, 359, 404, 942, 175, 428, 752, 602, 422, 978, 874, 395, 547, 906, 439, 521, 255,
                            134, 207, 825, 498, 687, 656, 945, 401, 326, 806, 581, 83, 130, 668, 697, 286, 718, 955,
                            374, 283, 3, 807, 414, 34, 290, 56, 1005, 762, 896, 951, 269, 791, 803, 163, 853, 27, 456,
                            114, 177, 267, 391, 824, 79, 1013, 345, 334, 713, 858, 358, 789, 332, 876, 292, 484, 204,
                            593, 369, 85, 412, 729, 596, 138, 198, 170, 103, 543, 65, 306, 947, 430, 487, 392, 787, 150,
                            479, 429, 934, 133, 703, 385, 89, 37, 949, 176, 709, 704, 6, 18, 393, 63, 855, 327, 467,
                            398, 815, 309, 95, 1016, 86, 433, 442, 536, 777, 486, 435, 1002, 739, 1, 726, 388, 39, 102,
                            973, 804, 379, 679, 450, 424, 482, 226, 827, 724, 21, 919, 882, 811, 534, 847, 943, 187,
                            244, 598, 975, 382, 552, 573, 389, 531, 635, 35, 438, 904, 623, 657, 195, 592, 469, 106,
                            946, 387, 678, 475, 144, 702, 694]

    # 新建一个1024，256存放鲁棒性codebook
    new_codebook = torch.zeros(1024, 256)
    count = 0
    for index in rubust_decimal_index:
        new_codebook[index] = codebook[count]
        count += 1

    vqgan.quantize.embedding.weight = torch.nn.Parameter(new_codebook)

    # ------------------------ input & output ------------------------
    # 攻击强度
    gaussian_var = 0.005
    jpeg_qf = 90
    results_root = f'results_{jpeg_qf}_{gaussian_var}/'
    os.mkdir(results_root)

    w = args.fidelity_weight
    if args.input_path.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')):  # input single img path
        input_img_list = [args.input_path]

    else:  # input img folder
        if args.input_path.endswith('/'):  # solve when path ends with /
            args.input_path = args.input_path[:-1]
        # scan all the jpg and png images
        input_img_list = sorted(glob.glob(os.path.join(args.input_path, '*.[jpJP][pnPN]*[gG]')))
        # random.shuffle(input_img_list)

    # 多张图片的PSNR平均结果
    psnr_mean = [0, 0, 0]
    psnr_mean_correction = [0, 0, 0]

    ssim_mean = [0, 0, 0]
    ssim_mean_correction = [0, 0, 0]

    # -------------------- generating stego ---------------------
    for i, img_path in enumerate(input_img_list):

        # 读取secret
        img_name = os.path.basename(img_path)
        basename, ext = os.path.splitext(img_name)
        os.mkdir(os.path.join(results_root, basename))

        print(f'[{i + 1}/{len(input_img_list)}]Secret Processing: {img_name}')
        secret_img = cv2.imread(img_path)
        secret_img_t = img2tensor(secret_img / 255., bgr2rgb=True, float32=True)
        normalize(secret_img_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        secret_img_t = secret_img_t.unsqueeze(0).to(device)

        txt_path = os.path.join(results_root, basename, f'{basename}_info.txt')
        txt_file = open(txt_path, 'w')

        # secret输入VQGAN的encoder，获得256个indice，将2560个indice比特嵌入到diffusion初始lanten的符号位中，重复5次，生成stego
        with torch.no_grad():
            """VQEncoder部分"""
            # encoder特征
            encoder_feat = vqgan.VQEncoder(secret_img_t)
            # 获得codebook的secret indices
            indice = vqgan.get_indice(encoder_feat)
            indice = indice.tolist()

            # 获得indices_flag
            indices_flag = utils.indices_flag_1024(indice)

            # 转换为2560个比特
            bits = [format(i1, '010b') for i1 in indice]
            binary_array = [int(bit) for bit in ''.join(bits)]

            """噪声生成"""
            prompt = 'highly detailed concept art of a sakura plum tree made with water, overgrowth, Makoto Shinkai'
            # prompt = 'a face of a young lady'

            noise = torch.randn([1, 4, 64, 64], device=device, dtype=torch.float64)

            # 保存noise到文件
            noise_name = os.path.join(results_root, basename, f'{basename}_noise.pkl')
            with open(noise_name, 'wb') as f:
                pickle.dump(noise, f)

            """
                嵌入indices_flag，嵌入indices比特
            """
            noise_modify, shuffled_index_list, index_list = utils.insert_bit_sorted_new_method(noise, binary_array)

            noise_modify = utils.insert_bit_sorted_new_method_1024(noise_modify, indices_flag, insert_time=0)

            """噪声生成图像部分"""
            # 用noise进行stego生成，在noise中嵌入了比特
            generate_image, init_noise1, image_latent1 = coupled_stablediffusion(prompt=prompt,
                                                                                 reverse=False,
                                                                                 run_baseline=True,
                                                                                 guidance_scale=7.5,
                                                                                 init_noise=noise_modify,
                                                                                 )
            stego_img = generate_image[0]

            # save stego_img
            stego_img_ = cv2.cvtColor(np.asarray(stego_img), cv2.COLOR_RGB2BGR)
            save_stego_name = f'{basename}_stego.png'
            save_stego_path = os.path.join(results_root, basename, save_stego_name)
            imwrite(stego_img_, save_stego_path)

            # save secret_img
            save_secret_name = f'{basename}_secret.png'
            save_secret_path = os.path.join(results_root, basename, save_secret_name)
            imwrite(secret_img, save_secret_path)

            torch.cuda.empty_cache()

        # -------------------- revealing secret ---------------------
        # 直接读取刚刚生成的stego
        stego_name = os.path.basename(save_stego_path)
        stego_img = Image.open(save_stego_path)

        # 攻击：高斯噪声
        stego_g_noisy = skimage.util.random_noise(np.array(stego_img), mode='gaussian', mean=0, var=(gaussian_var) ** 2)
        g_noisy_name = f'{basename}_stego_gaussian.png'
        save_g_noisy_path = os.path.join(results_root, basename, g_noisy_name)

        stego_g_noisy = Image.fromarray(np.uint8(stego_g_noisy * 255))
        stego_g_noisy = np.array(stego_g_noisy)
        stego_g_noisy = cv2.cvtColor(stego_g_noisy, cv2.COLOR_BGR2RGB)
        imwrite(stego_g_noisy, save_g_noisy_path)

        stego_g_noisy_img = Image.open(save_g_noisy_path)

        # 攻击：JPEG压缩======================================================================================
        jpeg_name = f'{basename}_stego_jpeg.jpg'
        save_jpeg_path = os.path.join(results_root, basename, jpeg_name)

        stego_img.save(save_jpeg_path, 'JPEG', quality=jpeg_qf)
        stego_jpeg_img = Image.open(save_jpeg_path)

        # stego进行逆向过程，恢复得到image_latent，对符号位进行提取秘密比特——→秘密比特恢复得到indice——→VQDecoder得到secret'
        stego_with_attack = [stego_jpeg_img, stego_g_noisy_img, stego_img]  # stego_jpeg_img,
        stego_with_attack_path = [save_jpeg_path, save_g_noisy_path, save_stego_path]  # save_jpeg_path,
        attack = [f'jpeg_{jpeg_qf}', f'gaussian_{gaussian_var}', 'origin']

        for i_attack, img_attack in enumerate(stego_with_attack):

            print(f'\n[{i_attack + 1}/{len(stego_with_attack)}]Stego Processing: {stego_with_attack_path[i_attack]}')
            print(f'\n[{i_attack + 1}/{len(stego_with_attack)}]Stego Processing: {stego_with_attack_path[i_attack]}',
                  file=txt_file)

            with torch.no_grad():
                """生成图像后恢复噪声部分"""
                init_noise2, image_latent2 = coupled_stablediffusion(prompt="",
                                                                     reverse=True,
                                                                     run_baseline=True,
                                                                     guidance_scale=1,
                                                                     init_image=img_attack,
                                                                     )

                """从恢复噪声中提取indice部分"""
                # 加载保存的noise
                with open(noise_name, 'rb') as f:
                    loaded_noise = pickle.load(f)

                # ======================================== 提取秘密indices ==============================================================
                binary_array_final = utils.extract_bit_sorted_new_method(init_noise2[0], loaded_noise,
                                                                         shuffled_index_list)

                bit_acc_rate = utils.calculate_acc_rate(binary_array, binary_array_final)

                print(f'{attack[i_attack]}_2560_Acc: {bit_acc_rate}\n')
                print(f'{attack[i_attack]}_2560_Acc: {bit_acc_rate}\n', file=txt_file)

                # ======================================== 提取秘密flag =================================================================
                indices_flag_final1 = utils.extract_bit_sorted_new_method_1024(init_noise2[0], loaded_noise,
                                                                               insert_time=0)

                bit_acc_rate = utils.calculate_acc_rate(indices_flag, indices_flag_final1)

                print(f'{attack[i_attack]}_1024_Acc: {bit_acc_rate}')
                print(f'{attack[i_attack]}_1024_Acc: {bit_acc_rate}', file=txt_file)

                # 比特表转十进制indice
                avg_indice = utils.convert_binary_to_decimal(binary_array_final, 10)

                # indice差异率
                indice_different_rate = utils.calculate_acc_rate(indice, avg_indice)
                print(f'{attack[i_attack]}_indice_Acc: {indice_different_rate}')
                print(f'{attack[i_attack]}_indice_Acc: {indice_different_rate}', file=txt_file)

                # indice差异列表
                indice_different_list = utils.calculate_difference_list(indice, avg_indice)
                print(f"差异列表: {indice_different_list}")
                print(f"差异列表: {indice_different_list}", file=txt_file)

                """从恢复的indice得到codebook向量，VQDecoder部分"""
                quant_feat = vqgan.get_quant_feat(avg_indice)
                secret_img_recon = vqgan.VQDecoder(quant_feat)
                secret_img_recon = tensor2img(secret_img_recon, rgb2bgr=True, min_max=(-1, 1))
                secret_img_recon = secret_img_recon.astype('uint8')

                # # save secret_img_recon
                save_recon_name = f'{basename}_recon_{attack[i_attack]}.png'
                save_restore_path = os.path.join(results_root, basename, save_recon_name)
                imwrite(secret_img_recon, save_restore_path)

                torch.cuda.empty_cache()

                # ====== 纠错编码，对应论文mapping2：根据indices_flag，查找恢复的indices是否在flag内，若不在，检查其翻转10次的indices在不在，若在则直接替代，若10个都不在，则维持 ==

                # 复制avg_indice的切片，存储纠错后的indice
                avg_indice_correction = avg_indice[:]

                for i2, index_ in enumerate(avg_indice_correction):
                    # 该index不在flag里面
                    if indices_flag_final1[index_] == 0:
                        # 产生10个翻转试错
                        flip_10 = utils.flip_bits(index_)
                        # 候选试错列表，用于挑选
                        candidate_flip_list = []
                        for flip_ in flip_10:
                            # 如果试错后的indice出现在flag中，则进行候选标记
                            if indices_flag_final1[flip_] == 1:
                                avg_indice_correction[i2] = flip_

                """从恢复的indice得到codebook向量，VQDecoder部分"""
                quant_feat = vqgan.get_quant_feat(avg_indice_correction)
                secret_img_recon_correction = vqgan.VQDecoder(quant_feat)
                secret_img_recon_correction = tensor2img(secret_img_recon_correction, rgb2bgr=True, min_max=(-1, 1))
                secret_img_recon_correction = secret_img_recon_correction.astype('uint8')

                # # save secret_img_recon
                save_recon_name_correction = f'{basename}_recon_correction_{attack[i_attack]}.png'
                save_restore_path_correction = os.path.join(results_root, basename, save_recon_name_correction)
                imwrite(secret_img_recon_correction, save_restore_path_correction)

                torch.cuda.empty_cache()

            secret_img = cv2.imread(img_path)
            stego_img_ = cv2.imread(stego_with_attack_path[i_attack])

            # 确定拼接后图像的尺寸
            height = max(secret_img.shape[0], stego_img_.shape[0], secret_img_recon.shape[0])
            width = secret_img.shape[1] + stego_img_.shape[1] + secret_img_recon.shape[1]

            # 创建一个空白图像作为拼接结果
            result = np.zeros((height, width, 3), dtype=np.uint8)

            # 将三张图片拷贝到拼接结果中
            result[:secret_img.shape[0], :secret_img.shape[1]] = secret_img
            result[:stego_img_.shape[0], secret_img.shape[1]:secret_img.shape[1] + stego_img_.shape[1]] = stego_img_
            result[:secret_img_recon.shape[0], secret_img.shape[1] + stego_img_.shape[1]:] = secret_img_recon

            save_all_name = f'{basename}_all_{attack[i_attack]}.png'
            save_all_path = os.path.join(results_root, basename, save_all_name)
            imwrite(result, save_all_path)

            # 存correct拼接结果
            # 创建一个空白图像作为拼接结果
            result = np.zeros((height, width, 3), dtype=np.uint8)

            # 将三张图片拷贝到拼接结果中
            result[:secret_img.shape[0], :secret_img.shape[1]] = secret_img
            result[:stego_img_.shape[0], secret_img.shape[1]:secret_img.shape[1] + stego_img_.shape[1]] = stego_img_
            result[:secret_img_recon_correction.shape[0],
            secret_img.shape[1] + stego_img_.shape[1]:] = secret_img_recon_correction

            save_all_name = f'{basename}_all_correction_{attack[i_attack]}.png'
            save_all_path = os.path.join(results_root, basename, save_all_name)
            imwrite(result, save_all_path)

            # cal psnr-/ssim
            psnr = calculate_psnr(secret_img_recon, secret_img, 0)
            ssim = calculate_ssim(secret_img_recon, secret_img, 0)
            print(f'【No Correction】    PSNR: {psnr}, SSIM: {ssim}')
            print(f'【No Correction】    PSNR: {psnr}, SSIM: {ssim}', file=txt_file)
            psnr_mean[i_attack] += psnr
            ssim_mean[i_attack] += ssim

            print('')
            print('', file=txt_file)
            # 计算纠错后的indice差异率
            indice_different_rate = utils.calculate_acc_rate(indice, avg_indice_correction)
            print(f'{attack[i_attack]}_indice_Acc: {indice_different_rate}')
            print(f'{attack[i_attack]}_indice_Acc: {indice_different_rate}', file=txt_file)

            # 纠错后的indice差异列表
            indice_different_list = utils.calculate_difference_list(indice, avg_indice_correction)
            print(f"差异列表: {indice_different_list}")
            print(f"差异列表: {indice_different_list}", file=txt_file)

            # cal psnr-/ssim
            psnr = calculate_psnr(secret_img_recon_correction, secret_img, 0)
            ssim = calculate_ssim(secret_img_recon_correction, secret_img, 0)
            print(f'【After Correction】 PSNR: {psnr}, SSIM: {ssim}')
            print(f'【After Correction】 PSNR: {psnr}, SSIM: {ssim}', file=txt_file)
            psnr_mean_correction[i_attack] += psnr
            ssim_mean_correction[i_attack] += ssim

            print('\n-----------------------------------------------------------------------------------------------\n')
            print(
                '\n-----------------------------------------------------------------------------------------------\n',
                file=txt_file)

        # 输出多张图像平均的PSNR结果
        print(f'\n前{i + 1}张图像平均的PSNR平均结果：')
        print(f'【 JPEG/ Gaussian/ Origin 】：          \n', [psnr / (i + 1) for psnr in psnr_mean],
              [ssim / (i + 1) for ssim in ssim_mean])
        print(f'【 JPEG/ Gaussian/ Origin 】：CORRECTION\n', [psnr / (i + 1) for psnr in psnr_mean_correction],
              [ssim / (i + 1) for ssim in ssim_mean_correction])

        print(f'\n前{i + 1}张图像平均的PSNR平均结果：',
              file=txt_file)
        print(f'【 JPEG/ Gaussian/ Origin 】：          \n', [psnr / (i + 1) for psnr in psnr_mean],
              [ssim / (i + 1) for ssim in ssim_mean],
              file=txt_file)
        print(f'【 JPEG/ Gaussian/ Origin 】：CORRECTION\n', [psnr / (i + 1) for psnr in psnr_mean_correction],
              [ssim / (i + 1) for ssim in ssim_mean_correction],
              file=txt_file)

        print('\n-----------------------------------------------------------------------------------------------')
        print(
            '\n-----------------------------------------------------------------------------------------------',
            file=txt_file)

        txt_file.close()
