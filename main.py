#coding:utf-8
import os
import sys
import cv2
import time
import scipy
import logging
import numpy as np
from scipy import sparse
import scipy.linalg as sl
from time import localtime, strftime
from scipy.sparse import csc_matrix, csr_matrix, bmat
from scipy.sparse import csc_matrix, linalg as sla
from scipy.optimize import nnls
from datetime import datetime
from functools import wraps
from contribute import getDefaultAffineGridSize,buildAffineSliceMatrix, \
						buildApplyAffineModelMatrix,buildDerivYMatrix,buildDerivXMatrix,\
						buildSecondDerivZMatrix,bguSlice

def get_run_time(func):
    def wrapper():
        begin_t = datetime.datetime.now()
        func()
        end_t   = datetime.datetime.now()
        logging.info('[INFO]({})>>> run time'.format(current_time(),(begin_t - end_t).seconds))
    return wrapper


def current_time():
    return strftime("%Y-%m-%d %H:%M:%S", localtime())

def get_full_dir(dir_path, pic_name):
    path_ = os.path.join(dir_path, pic_name)
    return path_

def im2double(input_ndarray):
    try:
        # im1 = input_ndarray[:,:,0]
        # input_ndarray[:,:,0] = input_ndarray[:,:,2]
        # input_ndarray[:,:,2] = im1
        input_ndarray = cv2.cvtColor(input_ndarray, cv2.COLOR_BGR2RGB)
        min_val = np.min(input_ndarray.ravel())
        max_val = np.max(input_ndarray.ravel())
        out = (input_ndarray.astype('float') - min_val) / float(max_val - min_val)
        return out
    except Exception, e:
        logging.error('[INFO]>>> 转双精度失败:{}'.format(e))
        sys.exit()

# def im2double(im):
#     im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
#     info = np.iinfo(im.dtype)
#     logging.info('[INFO]>>> info.max:{}'.format(info.max)) 
#     return im.astype(np.float) / info.max 

# @get_run_time
def rgb2luminance(rgb, coeffs=None):
    if(isinstance(rgb, np.ndarray)):
        if(coeffs is None):
            coeffs = np.array([0.25, 0.5, 0.25])

        if(len(rgb.shape) != 3):
            logging.error('[INFO]>>> rgb should be height x width x 3')
            return None
        if(len(coeffs) != 3):
            logging.error('[INFO]>>> coeffs must be a 3-element vector')
            return None
        if(abs(1 - np.sum(coeffs)) > 1e-6):
            logging.warning('[INFO]>>> coeffs sum to {}, which is not 1'.format(np.sum(coeffs)))
        try:
            luma = coeffs[0] * rgb[:,:,0] + coeffs[1] * rgb[:,:,1] + coeffs[2] * rgb[:,:,2]
            # logging.info('[INFO]>>> luma:{}'.format(luma))
            logging.info(u'[INFO]>>> rgb转luminance成功')
        except Exception, e:
            logging.error('[INFO]>>> rgb2luminance error:{}'.format(e))
            return None
        return luma

# @get_run_time
def testBGU(input_ds, edge_ds, output_ds, input_fs, edge_fs):
    output_dict = dict()
    output_dict['input_ds']   = input_ds
    output_dict['edge_ds']    = edge_ds
    output_dict['output_ds']  = output_ds
    output_dict['input_fs']   = input_fs
    output_dict['edge_fs']    = edge_fs

    output_dict['weight_ds']  = np.array([],dtype=np.float64)
    output_dict['grid_size']  = np.array([],dtype=np.float64)
    output_dict['lambda_s']   = None
    output_dict['intensity_options']   = dict()

    logging.info(u'[INFO]>>> 进入bguFit处理')
    output_dict['gamma'] = \
    bguFit(input_ds, edge_ds, output_ds, output_dict['weight_ds'], output_dict['grid_size'], \
        output_dict['lambda_s'], output_dict['intensity_options'])

    output_dict['grid_size'] = output_dict['gamma'].shape

    result = bguSlice(output_dict['gamma'], input_fs, edge_fs)

    #output['result_fs'] = bguSlice(output['gamma'], input_fs, edge_fs)

    return result



def showTestResults(tr, output_dir):
    cv2.imwrite(tr.result_fs,output_dir)


def get_td_list(width, height):
    if(width < 0 or height < 0):
        logging.error('[INFO]>>> width or height error')
        return False
    else:
        return [[0 for i in range(height)] for j in range(width)]

def bguFit(input_image, edge_image, output_image, output_weight=[], \
            grid_size=[], lambda_spatial=None, intensity_options={}):
    DEFAULT_LAMBDA_SPATIAL = 1
    DEFAULT_FIRST_DERIVATIVE_LAMBDA_Z = 4e-6
    DEFAULT_SECOND_DERIVATIVE_LAMBDA_Z = 4e-7

    if(len(output_weight) == 0):
    	logging.info(u'[INFO]>>> 权重矩阵output_weight获取默认')
        output_weight = np.ones(output_image.shape)

    if(len(grid_size) == 0):
        logging.info(u'[INFO]>>> 仿射网格grid_size获取默认大小')
        grid_size = getDefaultAffineGridSize(input_image, output_image)

    if(lambda_spatial is None):
    	logging.info(u'[INFO]>>> lambda_spatial获取默认大小')
        lambda_spatial = DEFAULT_LAMBDA_SPATIAL

    if(len(intensity_options) == 0):
        logging.info(u'[INFO]>>> 强度选项intensity_options获取默认大小')
        intensity_options['type']   = 'second'
        intensity_options['value']  = 0
        intensity_options['lambda'] =  DEFAULT_SECOND_DERIVATIVE_LAMBDA_Z
        intensity_options['enforce_monotonic'] = False

    input_height = input_image.shape[0]
    input_width = input_image.shape[1]

    grid_height = grid_size[0]
    grid_width = grid_size[1]
    grid_depth = grid_size[2]
    affine_output_size = grid_size[3]
    affine_input_size = grid_size[4]

    bin_size_x = input_width / grid_width
    bin_size_y = input_height / float(grid_height)
    bin_size_z = 1 / float(grid_depth)

    # 数据核对
    # logging.info('[INFO]>>> b_s_x:{}\n b_s_y:{}\n b_s_z:{}\n'.format(bin_size_x,bin_size_y,bin_size_z))

    num_deriv_y_rows = (grid_height - 1) * grid_width * grid_depth \
    * affine_output_size * affine_input_size
    num_deriv_x_rows = grid_height * (grid_width - 1) * grid_depth \
    * affine_output_size * affine_input_size

    weight_matrices = get_td_list(affine_output_size, affine_input_size)
    slice_matrices  = get_td_list(affine_output_size, affine_input_size)

    # weight_matrices = np.zeros(shape=(affine_output_size,affine_input_size))
    # slice_matrices  = np.zeros(shape=(affine_output_size,affine_input_size))
    # logging.info('[INFO]>>> affine_input_size:{}  affine_output_size:{}'.format(affine_input_size, affine_output_size))  
    for j in xrange(1,affine_input_size+1):
        for i in xrange(1,affine_output_size+1):
            logging.info('[INFO]>>> Building weight and slice matrices, i = {}, j = {}' \
                .format(i,j))
            [weight_matrices[i-1][j-1], slice_matrices[i-1][j-1]] = buildAffineSliceMatrix(\
                input_image, edge_image, grid_size, i, j)
    # 数据核对√
    # logging.info('[INFO]>>> weight_matrices:{}\n slice_matrices:{}\n'.format(weight_matrices[0][0],slice_matrices[0][0]))
    slice_matrix  = None
    weight_matrix = None

    for j in xrange(1,affine_input_size+1):
        for i in xrange(1,affine_output_size+1):
            # logging.info('[INFO]>>> Concatenating affine slice matrices, i = {}, j = {} \n' \
                # .format(i, j))
            # np.vstack 垂直组合
            # logging.info('[INFO]>>> slice_matrix.shape:{}'.format(slice_matrix.shape))
            if(slice_matrix is None and weight_matrix is None):
            	slice_matrix = slice_matrices[i-1][j-1]
            	weight_matrix = weight_matrices[i-1][j-1]
            else:
                slice_matrix  = bmat([[slice_matrix], [slice_matrices[i-1][j-1]]])
                # weight_matrix = sl.block_diag(weight_matrix, weight_matrices[i-1][j-1])
                weight_matrix = bmat([[weight_matrix,None],[None,weight_matrices[i-1][j-1]]])
    # 数据核对√
    logging.info('[INFO]>>> weight_matrix:{}\n slice_matrix:{}\n'.format(weight_matrix,slice_matrix))
    logging.info('Building apply affine model matrix\n')
    apply_affine_model_matrix = buildApplyAffineModelMatrix( \
        input_image, affine_output_size)
    # logging.info('[INFO]>>> a_a_m_m:{}\n'.format(apply_affine_model_matrix))
    # 数据核对√

    logging.info('Building full slice matrix\n')
    sqrt_w = np.sqrt(output_weight.flatten())
    sqrt_w = np.mat(sqrt_w).T
    logging.info('[INFO]>>> sqrt_w.shape:{} sqrt_w:{}'.format(sqrt_w.shape, sqrt_w))
    W_data = sparse.spdiags(sqrt_w.flatten(), 0, np.size(output_weight), np.size(output_weight))
    # logging.info('[INFO]>>> w_data:{} a_m{} w_m:{} s_m:{}'.format(\
    # 	W_data,apply_affine_model_matrix,weight_matrix, slice_matrix))
    logging.info('[INFO]>>> w_data.shape:{}\n w_data:{}'.format(W_data.shape, W_data))

    A_data = W_data * apply_affine_model_matrix * weight_matrix * slice_matrix
    # logging.info('[INFO]>>> A_data.type:{} \n A_data:{}'.format(type(A_data),A_data))

    # logging.info('[INFO]>>> output_image:{}\n'.format(output_image[0:10,0:10,0]))
    # b_data = np.multiply(output_image.T.flatten(), sqrt_w)
    # b_data = np.mat(b_data).T
    b_data = np.vstack((output_image[:,:,0].T.flatten(),output_image[:,:,1].T.flatten()))
    b_data = np.vstack((b_data,output_image[:,:,2].T.flatten())).flatten()
    b_data = np.mat(b_data).T
    logging.info('[INFO]>>> b_data.shape:{}\n b_data:{}\n'.format(b_data.shape,b_data[0:20 ]))

    # 数据核对√
    # logging.info('[INFO]>>> b_data.shape:{}\n b_data:{}\n'.format(b_data.shape,b_data))

    logging.info('Building d/dy matrix\n')
    A_deriv_y = (bin_size_x * bin_size_z / bin_size_y) * lambda_spatial * \
        buildDerivYMatrix(grid_size)
    b_deriv_y = np.zeros((num_deriv_y_rows, 1))

    logging.info('Building d/dx matrix\n')
    A_deriv_x = (bin_size_y * bin_size_z / bin_size_x) * lambda_spatial * \
        buildDerivXMatrix(grid_size)
    b_deriv_x = np.zeros((num_deriv_x_rows, 1))
    # logging.info('[INFO]>>> A_d_y:{}\n A_d_x:{}\n'.format(A_deriv_y,A_deriv_x))

    logging.info('Building d/dz matrix\n')
    if(intensity_options['type'] == 'second'):
        # logging.info('[INFO]>>> b_x * b_y:{}  b_z * b_z:{}'.format((bin_size_x * bin_size_y) , (bin_size_z * bin_size_z)))
        A_intensity = (bin_size_x * bin_size_y) / (bin_size_z * bin_size_z) * \
            intensity_options['lambda'] * buildSecondDerivZMatrix(grid_size)
        value = intensity_options['lambda'] * intensity_options['value']
        m = A_intensity.shape[0]
        b_intensity = value * np.ones((m, 1))

    # logging.info('[INFO]>>> A_d:{}\n A_d_y:{}\n A_d_x:{}\n A_i:{}\n'.format(A_data, \
    # 	A_deriv_y, A_deriv_x, A_intensity))
    # logging.info('[INFO]>>> b_d:{}\n b_d_y:{}\n b_d_x:{}\n b_i:{}\n'.format(b_data, b_deriv_y, b_deriv_x, b_intensity))
    A = bmat([[A_data], [A_deriv_y], [A_deriv_x]], format='csr')
    A = bmat([[A],[A_intensity]], format='csr')
    # A = np.vstack((A_data,A_deriv_y,A_deriv_x,A_intensity))
    # b = bmat([[b_data], [b_deriv_y], [b_deriv_x]], format='csr')
    # b = bmat([[b],[b_intensity]], format='csr')
    b = np.vstack((b_data, b_deriv_y, b_deriv_x, b_intensity))
    b = np.array(b)
    b = b.reshape((-1,))
    # 数据核对√
    logging.info('[INFO]>>> A.shape:{} b.shape:{}'.format(A.shape, b.shape))
    logging.info('[INFO]>>> type(A):{} type(b):{}'.format(type(A), type(b)))
    logging.info('[INFO]>>> A:{} \n b:{}'.format(A,b))

    logging.info('Solving system\n')
    # 对应于MATLAB中 inv() 函数, 矩阵求逆
    # A = np.linalg.inv(A)
    # A1 = sparse.linalg.inv(A)
    # gamma = np.linalg.solve(A, b)
    # gamma = sparse.linalg.lstsq(A,b)
    linalg_out  = sparse.linalg.lsmr(A, b)
    gamma = linalg_out[0]

    # gamma = sparse.linalg.spsolve(A.T * A, A.T * b)
    logging.info('[INFO]>>> gamma:{}'.format(gamma))
    # gamma = nnls(A1, b)
    gamma_ = gamma.reshape(grid_size)	
    return gamma_
    # return [gamma, A, b, lambda_spatial, intensity_options]


def main(pic_dir, low_res_in_pic, low_res_out_pic, high_res_in_pic):

    # 获取高分辨率原图
    logging.info(u'[INFO]>>> 高分辨率原图转双精度\n')
    input_fs = im2double(cv2.imread(get_full_dir(pic_dir, high_res_in_pic)))
    logging.info('[INFO]>>> input_fs:{}'.format(input_fs[:,:,0]))
    logging.info(u'[INFO]>>> rgb转图像亮度\n')
    edge_fs  = rgb2luminance(input_fs)
    logging.info('[INFO]>>> egde_fs.shape:{} \n edge_fs:{}'.format(edge_fs.shape, edge_fs))
    if(edge_fs is None):
        logging.error(u'[INFO]>>> 高分辨率原图rgb转亮度失败\n')
        sys.exit()

    # 获取低分辨率原图
    logging.info(u'[INFO]>>> 低分辨率原图转双精度\n')
    input_ds = im2double(cv2.imread(get_full_dir(pic_dir, low_res_in_pic)))
    logging.info('[INFO]>>> input_ds:{}'.format(input_ds[:,:,0]))
    logging.info(u'[INFO]>>> rgb转图像亮度\n')
    edge_ds = rgb2luminance(input_ds)
    logging.info('[INFO]>>> edge_ds.shape:{} \n edge_ds:{}'.format(edge_ds.shape, edge_ds))
    # logging.info('[INFO]>>> input_ds:{}'.format(input_ds[0:10,0:10,2]))
    if(edge_ds is None):
        logging.error(u'[INFO]>>> 低分辨率原图rgb转亮度失败\n')
        sys.exit()

    # 获取低分辨率输出图
    logging.info(u'[INFO]>>> 低分辨率输出图转双精度\n')
    output_ds = im2double(cv2.imread(get_full_dir(pic_dir, low_res_out_pic)))
    logging.info('[INFO]>>> output.shape:{} \n output_ds:{}'.format(output_ds.shape, output_ds[0:10,0:10,0]))

    logging.info(u'[INFO]>>> 开始进入bgu处理\n')
    result = testBGU(input_ds, edge_ds, output_ds, input_fs, edge_fs)
    # showTestResults(result)
    cv2.imwrite('./pic/output.jpg',result)


if __name__ == "__main__":

    if(len(os.listdir('./logs/')) > 0):
        os.system('sudo rm ./logs/*')
    logging.basicConfig(
            filename='logs/bgu_pro' + time.strftime('%Y-%m-%d', time.localtime(time.time())) + '.log',
            level=logging.INFO, format="[%(asctime)s - %(message)s")

    logging.info(u'[INFO]>>> bgu主程序启动\n')
    main('./pic', 'low_res_in.png', 'low_res_out.png', 'high_res_in.png')
