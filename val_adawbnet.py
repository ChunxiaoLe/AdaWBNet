import ntpath
import torch
import os
import glob
from datetime import datetime
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from skimage.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import argparse
# import models.StyTR  as StyTR
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os
from err import err_compution,error_evaluation,err_compution_npy
import sys
import models.StyTR_OR as StyTR

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loader = transforms.Compose([transforms.ToTensor()])  # transform it into a torch tensor

unloader = transforms.ToPILImage()  # reconvert into PIL image

parser = argparse.ArgumentParser()
"BSET: BASE_Trans_3_8_256_OR_awl3_bz4"
parser.add_argument('--model_type', default='BASE_Trans_3_8_256_OR_awl3_bz4', type=str)
parser.add_argument('--dataset_type', default='cube1', type=str)
"/home/wsy/folds/jint test_cube_.npy"
"/home/wsy/sty2/folds/test_set1.npy or set2.npy"
parser.add_argument('--dataset_dir', default='/home/wsy/sty2/folds/test_cube.npy', type=str)
parser.add_argument('--numcluster', default=256, type=int)
args = parser.parse_args()

# 加载时间最近的模型文件
# 加载时间最近的模型文件
def load_latest_model_by_type(model_dir, model, model_type):
    pattern = f"*{model_type}*"
    model_files = glob.glob(os.path.join(model_dir, pattern))
    latest_model = max(model_files, key=os.path.getctime)
    model_dict = torch.load(latest_model)
    model.load_state_dict(model_dict)
    print(f"Loaded {model_type} model: {latest_model}")
    return model


def image_loader(image_name):
    image0 = Image.open(image_name)
    image_resize = loader(image0.resize((256, 256))).unsqueeze(0)
    image = loader(image0).unsqueeze(0)
    return image_resize.to(device, torch.float), image.to(device, torch.float)

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    plt.axis('off')
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def apply_mapping_func(image, m):
    """ Applies the polynomial mapping """
    sz = image.shape
    image = np.reshape(image, [-1, 3])
    result = m.predict(kernelP(image))
    result = np.reshape(result, [sz[0], sz[1], sz[2]])
    return result

def kernelP(I):
    """ Kernel function: kernel(r, g, b) -> (r,g,b,rg,rb,gb,r^2,g^2,b^2,rgb,1)
        Ref: Hong, et al., "A study of digital camera colorimetric characterization
         based on polynomial modeling." Color Research & Application, 2001. """
    return (np.transpose((I[:, 0], I[:, 1], I[:, 2], I[:, 0] * I[:, 1], I[:, 0] * I[:, 2],
                          I[:, 1] * I[:, 2], I[:, 0] * I[:, 0], I[:, 1] * I[:, 1],
                          I[:, 2] * I[:, 2], I[:, 0] * I[:, 1] * I[:, 2],
                          np.repeat(1, np.shape(I)[0]))))

def get_mapping_func(image1, image2):
    """ Computes the polynomial mapping """
    image1 = np.reshape(image1, [-1, 3])
    image2 = np.reshape(image2, [-1, 3])
    m = LinearRegression().fit(kernelP(image1), image2)
    return m

def outOfGamutClipping(I):
    """ Clips out-of-gamut pixels. """
    I[I > 1] = 1  # any pixel is higher than 1, clip it to 1
    I[I < 0] = 0  # any pixel is below 0, clip it to 0
    return I

# 加载数据集
def load_test_data(data_dir):
    # 假设数据是 numpy 数组或者图像，进行相应的处理
    # 此处示例使用随机生成数据代替
    files = np.load(data_dir,
                    allow_pickle=True).item()
    x_test_list = files['input']
    y_test_list = files['awb']

    # x_dir = '/dataset/set2/Set2_input_images/'
    # y_dir = '/dataset/set2/Set2_gt_images/'
    # img_ex1 = 'DSLR_'
    # img_list1 = ['01009','01010','01011','01012','01013','01014','01016','01017','01020','01022','01023',
    #             '01024','01025','01026','01027','01028',
    #             '01111','01113','01114','01115','01116','01117','01118','01414','01415','01416','01417',
    #             '01418','01419','01426','01427','01428','01429',
    #             '01470','01471','01472','01475','01477','01479','01480','01482','01483','01484','01485',
    #             '01625','01626','01627','01629','01630','01632','01634','01635',
    #              '00955','00952']
    # img_ex2 = 'Mobile_'
    # img_list2 = ['00798','00799','00800','00803','00805','00806','00807','00808','00809','00810',
    #              '00813','008134','00815','00816','00818','00819','00820','00822','00823','00824','00825',
    #              '00868','00869','00870','00872','00873','00874',
    #              '00973','00974','00975','00976','00977','00978','00981','00982','00983','00984','00985','00986',
    #              '00994','00996','00997','00998','01000',
    #              '00801','00802']
    # img_w = '.png'
    #
    # x_test_list, y_test_list = [],[]
    # for name in img_list1:
    #     x_test_list.append(x_dir + img_ex1 + name + img_w)
    #     y_test_list.append(y_dir + img_ex1 + name + img_w)
    #
    # for name in img_list2:
    #     x_test_list.append(x_dir + img_ex2 + name + img_w)
    #     y_test_list.append(y_dir + img_ex2 + name + img_w)
    #
    # x_test_list = np.array(x_test_list)
    # y_test_list = np.array(y_test_list)

    # x_test_list = files['name']
    # y_test_list = files['gt']
    # x_test_list = np.array([path.replace('/dataset/lcx/cube/Cube_input_images/', '/dataset/cube/input/') for path in x_test_list])
    # y_test_list = np.array([path.replace('/dataset/cube/gt//', '/dataset/cube/gt/') for path in y_test_list])
    # #
    # save_ = {}
    # save_['input'] = x_test_list
    # save_['awb'] = y_test_list
    # np.save('/home/wsy/sty2/folds/test_set2_multi.npy', save_)

    return x_test_list, y_test_list

def imsave_npy(image_npy,title):
    # image_npy = np.clip(image_npy, 0, 255)  # 如果数据范围不在0到255之间，需要先裁剪
    # image_npy = image_npy.astype(np.uint8)  # 将数据类型转换为 uint8
    image_npy = Image.fromarray((image_npy*255).astype(np.uint8))
    # image = Image.fromarray(image_npy)
    image_npy.save(title)

def imsave(tensor, title):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    image.save(title)
    # plt.imshow(image)
    # plt.axis('off')
    # plt.show()
    # if title is not None:
    #     plt.title(title)
    # plt.pause(0.001)  # pause a bit so that plots are updated

def unloader_tensor(tensor):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = np.array(unloader(image))/255
    return image

# 测试模型
def test_model(transformer, embedding, decoder, x_test_list, y_test_list, save_dir, result_file):
    transformer.to(device).eval()
    embedding.to(device).eval()
    decoder.to(device).eval()
    # num_img = X_test_list.shape
    DE, MSE, MAE = [], [], []
    with torch.no_grad():
        # x_test_list.shape[0]
        for num in range(x_test_list.shape[0]):
            if os.path.exists(x_test_list[num]) and os.path.exists(y_test_list[num]):
                print(x_test_list[num],'--', y_test_list[num])

                filename = os.path.split(x_test_list[num])[-1]
                input, input_full = image_loader(x_test_list[num])
                gt, gt_full = image_loader(y_test_list[num])

                "测试"
                pos_s = None
                pos_c = None
                mask = None
                style = embedding(input)
                content = embedding(input)
                hs = transformer(style, mask, content, pos_c, pos_s)
                esti = decoder(hs)
                esti = torch.clip(esti,0,1)
                esti_npy = unloader_tensor(esti)
                input_npy = unloader_tensor(input)
                input_full_npy = unloader_tensor(input_full)
                m_awb1 = get_mapping_func(input_npy, esti_npy)
                output_awb = outOfGamutClipping(apply_mapping_func(input_full_npy, m_awb1))

                # imsave_npy(output_awb, save_dir + filename)
                # "可视化"
                # plt.imshow(output_awb)
                # plt.show()
                # imshow(esti)

                "测试指标"
                "Delta差异"
                gt_full_npy = unloader_tensor(gt_full)
                deltae, mse, mae = err_compution_npy(gt_full_npy, output_awb)
                DE.append(deltae)
                MSE.append(mse[0])
                MAE.append(mae)

    DE = np.array(DE)
    MSE = np.array(MSE)
    MAE = np.array(MAE)

    print('-------------DELTAE-------------')
    Mean_DE,x_DE,y_DE,z_DE = error_evaluation(DE)
    print('-------------MSE-------------')
    Mean_MSE,x_MSE,y_MSE,z_MSE = error_evaluation(MSE)
    print('-------------MAE-------------')
    Mean_MAE,x_MAE,y_MAE,z_MAE = error_evaluation(MAE)

    with open(result_file, 'w') as f:
        f.write(f"MSE-ALL: {Mean_MSE},{x_MSE},{y_MSE},{z_MSE}\n")
        f.write(f"MAE-ALL: {Mean_MAE},{x_MAE},{y_MAE},{z_MAE}\n")
        f.write(f"Delta E2000-ALL: {Mean_DE},{x_DE},{y_DE},{z_DE}\n")
    print(f"Test results saved to {result_file}")



    return DE, MSE, MAE

def test_model_whole(net, x_test_list, y_test_list, save_dir, result_file):
    net.to(device).eval()
    # num_img = X_test_list.shape
    DE, MSE, MAE = [], [], []
    with torch.no_grad():
        # x_test_list.shape[0]
        for num in range(x_test_list.shape[0]):
            if os.path.exists(x_test_list[num]) and os.path.exists(y_test_list[num]):
                print(x_test_list[num],'--', y_test_list[num])

                filename = os.path.split(x_test_list[num])[-1]
                input, input_full = image_loader(x_test_list[num])
                gt, gt_full = image_loader(y_test_list[num])

                "测试"
                pos_s = None
                pos_c = None
                mask = None
                esti = net(input)
                esti = torch.clip(esti,0,1)
                esti_npy = unloader_tensor(esti)
                input_npy = unloader_tensor(input)
                input_full_npy = unloader_tensor(input_full)
                m_awb1 = get_mapping_func(input_npy, esti_npy)
                output_awb = outOfGamutClipping(apply_mapping_func(input_full_npy, m_awb1))

                # imsave_npy(output_awb, save_dir + filename)
                # "可视化"
                # plt.imshow(output_awb)
                # plt.show()
                # # imshow(esti)

                "测试指标"
                "Delta差异"
                gt_full_npy = unloader_tensor(gt_full)
                deltae, mse, mae = err_compution_npy(gt_full_npy, output_awb)
                DE.append(deltae)
                MSE.append(mse[0])
                MAE.append(mae)

                # "按条件储存"
                # if deltae <= 5.5:
                #     imsave_npy(output_awb, save_dir + filename)

    DE = np.array(DE)
    MSE = np.array(MSE)
    MAE = np.array(MAE)

    print('-------------DELTAE-------------')
    Mean_DE,x_DE,y_DE,z_DE = error_evaluation(DE)
    print('-------------MSE-------------')
    Mean_MSE,x_MSE,y_MSE,z_MSE = error_evaluation(MSE)
    print('-------------MAE-------------')
    Mean_MAE,x_MAE,y_MAE,z_MAE = error_evaluation(MAE)

    with open(result_file, 'w') as f:
        f.write(f"MSE-ALL: {Mean_MSE},{x_MSE},{y_MSE},{z_MSE}\n")
        f.write(f"MAE-ALL: {Mean_MAE},{x_MAE},{y_MAE},{z_MAE}\n")
        f.write(f"Delta E2000-ALL: {Mean_DE},{x_DE},{y_DE},{z_DE}\n")
    print(f"Test results saved to {result_file}")

    return DE, MSE, MAE




# 主函数
def main(args):
    # 定义模型路径和数据集路径

    model_type = args.model_type
    dataset_type = args.dataset_type
    dataset_dir = args.dataset_dir
    model_dir = '/home/wsy/sty2/experiments/model/' + model_type +'/'


    save_dir = '/home/wsy/sty2/experiments/output/' + dataset_type + '/' + model_type + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    result_file = save_dir + 'test_results.txt'

    result_file_all = save_dir + 'test_results_all.txt'

    # 加载最新的模型
    # decoder = StyTR.decoder
    # embedding = StyTR.PatchEmbed()
    # "WithKMAX_10_14: num_cluster=5: +pos.?"
    # transformer = Trans.Transformer(num_cluster=args.numcluster)
    # transformer = load_latest_model_by_type(model_dir,transformer, "transformer")
    # embedding = load_latest_model_by_type(model_dir, embedding, "embedding")
    # decoder = load_latest_model_by_type(model_dir, decoder, "decoder")

    "统一加载模型"
    net = StyTR.StyTrans(device)
    print("--------loading checkpoint----------")
    checkpoint = torch.load(args.model_type + '/' + 'best_model.tar')
    net.load_state_dict(checkpoint['state_dict'])
    net.to(device)
    net.eval()

    # 加载测试数据
    "MixedScene Dataset, Rendered Cube Dataset, RenderedWB-Set2 Dataset, RenderedWB-Set1-Test Dataset"
    x_test_list, y_test_list = load_test_data(dataset_dir)

    # 测试模型并计算指标
    mse_val, mae_val, delta_e2000 = test_model_whole(net, x_test_list, y_test_list, save_dir, result_file)

    # 保存结果
    with open(result_file_all, 'w') as f:
        f.write(f"MSE: {mse_val}\n")
        f.write(f"MAE: {mae_val}\n")
        f.write(f"Delta E2000: {delta_e2000}\n")
    print(f"Test results saved to {result_file_all}")

    print('d')




if __name__ == '__main__':
    main(args)
