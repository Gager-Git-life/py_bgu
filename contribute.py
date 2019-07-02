#coding:utf-8
import os
import math
import scipy
# import numba
import logging
import numpy as np
import scipy.linalg as sl
from numpy.matlib import repmat
from scipy import sparse
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, bmat
from scipy.interpolate import RegularGridInterpolator as rgi

def get_td_list(width, height):
    if(width < 0 or height < 0):
        return False
    else:
        return [[None for _ in range(width)] for _ in range(height)]

def sub2ind(grid_size, st_bg_xx, st_bg_yy, st_bg_zz, st_bg_uu, st_bg_vv):
    # np.ravel_multi_index((st_bg_xx, st_bg_yy, st_bg_zz, st_bg_uu, st_bg_vv), \
    #     dims=grid_size)
    out = (st_bg_vv - 1) * (grid_size[0]*grid_size[1]*grid_size[2]*grid_size[3]) + \
          (st_bg_uu - 1) * (grid_size[0]*grid_size[1]*grid_size[2]) + \
          (st_bg_zz - 1) * (grid_size[0]*grid_size[1]) + \
          (st_bg_yy - 1) * (grid_size[0]) + \
          (st_bg_xx    )

    return out - 1

# @numba.jit(nopython=True)
def makeBoundaryZ1(grid_size, input_):
    mm = grid_size[0] * grid_size[1]
    nn = grid_size[0] * grid_size[1] * 2
    e  = np.ones((mm,1))
    B  = sparse.spdiags(np.hstack((-e, e)).T, [0, grid_size[0] * grid_size[1]], mm, nn) 
    # sparse.lil_matrix:创建一个空的稀疏矩阵
    return bmat([[B, sparse.lil_matrix((mm, input_ - nn))]], format='csr')

# @numba.jit(nopython=True)
def makeBoundaryZEnd(grid_size,input_):
    mm = grid_size[0] * grid_size[1]
    nn = grid_size[0] * grid_size[1]
    e  = np.ones((mm,1))
    B  = sparse.spdiags(np.hstack((e, -e)).T, [0, grid_size[0] * grid_size[1]], mm, nn)
    return bmat([[sparse.lil_matrix((mm, input_ - nn)),B]], format='csr')

# @numba.jit(nopython=True)
def buildAffineSliceMatrix(input_image, edge_image, grid_size, i, j):
    # np.prod()函数用来计算所有元素的乘积
    num_grid_cells = np.prod(grid_size)
    image_width = input_image.shape[1]
    image_height = input_image.shape[0]
    num_pixels = image_width * image_height
    grid_width = grid_size[1]
    grid_height = grid_size[0]
    grid_depth = grid_size[2]

    # logging.info('[INFO]>>> image_width:{} image_height:{}'.format(image_width, image_height))
    # logging.info('[INFO]>>> g_w:{} g_h:{} g_d:{}'.format(grid_width, grid_height, grid_depth))

    pixel_x = np.mat(np.arange(0,image_width))
    pixel_y = np.mat(np.arange(0,image_height)).T

    bg_coord_x = (pixel_x + 0.5) * (grid_width - 1) / image_width
    bg_coord_y = (pixel_y + 0.5) * (grid_height - 1) / image_height
    bg_coord_z = edge_image * (grid_depth - 1)
    # logging.info('[INFO]>>> bg_coord_z:{}'.format(bg_coord_z))

    # logging.info('[INFO]>>> bg_x:{} bg_y:{} bg_z:{}'\
    # 	.format(bg_coord_x.shape,bg_coord_y.shape,bg_coord_z.shape))
    # np.floor向下取整
    bg_idx_x0    = np.floor(bg_coord_x)
    bg_idx_y0    = np.floor(bg_coord_y)
    bg_idx_z0_im = np.floor(bg_coord_z)
    # logging.info('[INFO]>>> bg_idx_z0_im:{}'.format(bg_idx_z0_im))

    # np.tile函数的主要功能就是将一个数组重复一定次数形成一个新的数组
    bg_idx_x0_im = np.tile(bg_idx_x0, (image_height, 1))
    bg_idx_y0_im = np.tile(bg_idx_y0, (1, image_width))
    # logging.info('[INFO]>>> bg_idx_x0_im:{}'.format(bg_idx_x0_im))
    # logging.info('[INFO]>>> bg_idx_y0_im:{}'.format(bg_idx_y0_im))
    # logging.info('[INFO]>>> bg_idx_z0_im:{}'.format(bg_idx_z0_im))

    dx = np.tile(bg_coord_x - bg_idx_x0, (image_height, 1))
    dy = np.tile(bg_coord_y - bg_idx_y0, (1, image_width))
    dz = bg_coord_z - bg_idx_z0_im

    # logging.info('[INFO]>>> dx.shape:{} dy.shape:{} dz.shape:{}'.format(dx.shape,dy.shape,dz.shape))

    dx = np.asarray(dx)
    dy = np.asarray(dy)
    dz = np.asarray(dz)

    weight_000 = ((1 - dx) * (1 - dy) * (1 - dz)).T.flatten()
    weight_100 = ((    dx) * (1 - dy) * (1 - dz)).T.flatten()
    weight_010 = ((1 - dx) * (    dy) * (1 - dz)).T.flatten()
    weight_110 = ((    dx) * (    dy) * (1 - dz)).T.flatten()
    weight_001 = ((1 - dx) * (1 - dy) * (    dz)).T.flatten()
    weight_101 = ((    dx) * (1 - dy) * (    dz)).T.flatten()
    weight_011 = ((1 - dx) * (    dy) * (    dz)).T.flatten()
    weight_111 = ((    dx) * (    dy) * (    dz)).T.flatten()

    # logging.info('[INFO]>>> weight_000:{} \n weight_100:{}\n'.format(weight_000[0:10],weight_100[0:10]))
    # logging.info('[INFO]>>> weight_010:{} \n weight_110:{}\n'.format(weight_010[0:10],weight_110[0:10]))
    # logging.info('[INFO]>>> weight_001:{} \n weight_101:{}\n'.format(weight_001[0:10],weight_101[0:10]))
    # logging.info('[INFO]>>> weight_011:{} \n weight_111:{}\n'.format(weight_011[0:10],weight_111[0:10]))
    st_i = np.arange(0, (8 * num_pixels ))

    
    # bg_idx_x0_im = np.asarray(bg_idx_x0_im)
    bg_idx_x0_im = bg_idx_x0_im.T.flatten()
    # logging.info('[INFO]>>> bg_idx_x0_im:{}'.format(bg_idx_x0_im))
    st_bg_xx = np.vstack([(bg_idx_x0_im + 1), (bg_idx_x0_im + 2), \
                (bg_idx_x0_im + 1), (bg_idx_x0_im + 2), \
                (bg_idx_x0_im + 1), (bg_idx_x0_im + 2), \
                (bg_idx_x0_im + 1), (bg_idx_x0_im + 2)]).T
    # logging.info('[INFO]>>> st_bg_xx > 0:{}'.format(len(filter(lambda x:x>0, st_bg_xx))))
    # logging.info('[INFO]>>> st_bg_xx.shape:{}'.format(st_bg_xx.shape))

    # logging.info('[INFO]>>> bg_idx_y0_im.shape:{}'.format(bg_idx_y0_im.shape))
    # bg_idx_y0_im = np.asarray(bg_idx_y0_im)
    bg_idx_y0_im = bg_idx_y0_im.T.flatten()
    st_bg_yy = np.vstack([(bg_idx_y0_im + 1), (bg_idx_y0_im + 1), \
                (bg_idx_y0_im + 2), (bg_idx_y0_im + 2), \
                (bg_idx_y0_im + 1), (bg_idx_y0_im + 1), \
                (bg_idx_y0_im + 2), (bg_idx_y0_im + 2)]).T
    # logging.info('[INFO]>>> st_bg_yy > 0:{}'.format(len(filter(lambda x:x>0, st_bg_yy))))

    # logging.info('[INFO]>>> bg_idx_z0_im.shape:{}'.format(bg_idx_z0_im.shape))
    # bg_idx_z0_im = np.asarray(bg_idx_z0_im)
    bg_idx_z0_im = bg_idx_z0_im.T.flatten()
    # logging.info('[INFO]>>> bg_idx_z0_im:{}'.format(bg_idx_z0_im))
    st_bg_zz = np.vstack([(bg_idx_z0_im + 1), (bg_idx_z0_im + 1), \
                (bg_idx_z0_im + 1), (bg_idx_z0_im + 1), \
                (bg_idx_z0_im + 2), (bg_idx_z0_im + 2), \
                (bg_idx_z0_im + 2), (bg_idx_z0_im + 2)]).T
    # logging.info('[INFO]>>> st_bg_zz:{}'.format(st_bg_zz))
    # logging.info('[INFO]>>> st_bg_zz >0 <=grid_depth:{}'.format(len(filter(lambda x:(x>0)&(x<=grid_depth), st_bg_zz))))

    st_bg_uu = i * np.ones((8 * num_pixels, 1))
    st_bg_vv = j * np.ones((8 * num_pixels, 1))
    st_s     =     np.ones((8 * num_pixels, 1))

    # logging.info('[INFO]>>> st_bg_xx:{}'.format(st_bg_xx))
    # logging.info('[INFO]>>> st_bg_yy:{}'.format(st_bg_yy))
    # logging.info('[INFO]>>> st_bg_zz:{}'.format(st_bg_zz))
    # logging.info('[INFO]>>> st_bg_uu:{}'.format(st_bg_uu))
    # logging.info('[INFO]>>> st_bg_vv:{}'.format(st_bg_vv))

    indices  = ((st_bg_xx > 0) & (st_bg_xx <= grid_width)) \
        & ((st_bg_yy > 0) & (st_bg_yy <= grid_height)) \
        & ((st_bg_zz > 0) & (st_bg_zz <= grid_depth))
    # logging.info('[INFO]>>> indices:{}'.format(indices))

    indices_a  = np.asarray(indices).flatten()
    # logging.info('[INFO]>>> ST_BG_XX:{}'.format(len(filter(lambda x:(x>0)&(x<=grid_width), st_bg_xx))))
    # logging.info('[INFO]>>> ST_BG_YY:{}'.format(len(filter(lambda x:(x>0)&(x<=grid_height), st_bg_yy))))
    # logging.info('[INFO]>>> ST_BG_ZZ:{}'.format(len(filter(lambda x:(x>0)&(x<=grid_depth), st_bg_zz))))
    # logging.info('[INFO]>>> len indices:{}'.format(len(filter(lambda x:x == True, indices))))
    # logging.info('[INFO]>>> st_i:{}'.format(st_i.shape))
    st_i     = st_i[indices_a]
    st_bg_xx = st_bg_xx[indices]
    st_bg_yy = st_bg_yy[indices]
    st_bg_zz = st_bg_zz[indices]
    st_bg_uu = st_bg_uu[indices_a]
    st_bg_vv = st_bg_vv[indices_a]
    st_s     = st_s[indices_a].flatten()

    st_bg_xx = np.asmatrix(st_bg_xx).T
    st_bg_yy = np.asmatrix(st_bg_yy).T
    st_bg_zz = np.asmatrix(st_bg_zz).T

    # logging.info('[INFO]>>> grid_size:{}'.format(grid_size))
    # logging.info('[INFO]>>> st_bg_xx:{}'.format(st_bg_xx[0:10]))
    # logging.info('[INFO]>>> st_bg_yy:{}'.format(st_bg_yy[0:10]))
    # logging.info('[INFO]>>> st_bg_zz:{}'.format(st_bg_zz[0:10]))
    # logging.info('[INFO]>>> st_bg_uu:{}'.format(st_bg_uu[0:10]))
    # logging.info('[INFO]>>> st_bg_vv:{}'.format(st_bg_vv[0:10]))
    st_j = sub2ind(grid_size, st_bg_yy, st_bg_xx, st_bg_zz, st_bg_uu, st_bg_vv)
    st_j = np.asarray(st_j.T).flatten()
    st_m = 8 * num_pixels
    st_n = num_grid_cells

    logging.info('[INFO]>>> st_s.shape:{} st_i.shape:{} st_j.shape:{}'.format(st_s.shape, st_i.shape, st_j.shape))
    # logging.info('[INFO]>>> st_s:{} type(st_i):{} type(st_j):{}'.format(st_s[0:20], st_i[0:20], st_j[0:20]))
    logging.info('[INFO]>>> st_m:{} st_n:{}'.format(st_m, st_n))
    st  = csr_matrix((st_s, (st_i, st_j)), shape=(st_m, st_n))

    w_i = np.tile(np.arange(0,num_pixels), (8, 1)).T.flatten()
    w_j = np.arange(0,(8 * num_pixels))
    w_s = np.vstack([weight_000, weight_100, weight_010, weight_110, \
           weight_001, weight_101, weight_011, weight_111]).T.flatten()
    w_m = num_pixels
    w_n = 8 * num_pixels

    
    # logging.info('[INFO]>>> w_s:{} w_i:{} w_j:{}'.format(w_s.shape, w_i.shape, w_j.shape))
    # logging.info('[INFO]>>> w_s:{} w_i:{} w_j:{}'.format(w_s[0:10], w_i[0:10], w_j[0:10]))
    # logging.info('[INFO]>>> w_m:{} w_n:{}\n'.format(w_m, w_n))
    w   = csr_matrix((w_s, (w_i, w_j)), shape=(w_m, w_n))

    # logging.info('[INFO]>>> st.shape:{} w.shape:{}'.format(st.shape, w.shape))
    # logging.info('[INFO]>>> st:{} \n w:{}'.format(st[0:10][0:10],w[0:10][0:10]))
    return [w, st]

# @numba.jit(nopython=True)
def buildApplyAffineModelMatrix(input_image, num_output_channels):
    num_pixels = input_image.shape[0] * input_image.shape[1]
    A = None
    for k in xrange(input_image.shape[2]):
        plane = input_image[:,:,k]
        # logging.info('[INFO]>>> plane:{}\n'.format(np.size(plane) * num_output_channels))
        # logging.info('[INFO]>>> plane.flatten:{}\n'.format(plane.flatten()))
        sd = sparse.spdiags(np.tile(plane.T.flatten(), (num_output_channels, 1)).flatten(), \
            0, num_output_channels * num_pixels, num_output_channels * num_pixels)
        if(A is None):
        	A = sd
        	# logging.info('[INFO]>>> sd:{}'.format(sd.shape))
        else:
        	A = bmat([[A,sd]], format='csr')
        	# logging.info('[INFO]>>> A.shape:{}'.format(A.shape))

    ones_diag = sparse.spdiags(np.ones((num_output_channels * num_pixels, 1)).flatten(), 0, \
        num_output_channels * num_pixels, num_output_channels * num_pixels)


    # logging.info('[INFO]>>> A:{}  ones_diag:{}'.format(A, ones_diag))
    return bmat([[A, ones_diag]], format='csr')

# @numba.jit(nopython=True)
def buildDerivXMatrix(grid_size):
    m = grid_size[0] * (grid_size[1] - 1)
    n = grid_size[0] * grid_size[1]
    e = np.ones((m, 1))
    d_dx = sparse.spdiags(np.hstack((-e, e)).T, [0, grid_size[0]], m, n)	

    A = None

    for _ in xrange(0,grid_size[4]):
        for _ in xrange(0,grid_size[3]):
            for _ in xrange(0,grid_size[2]):
                if(A is None):
                    A = d_dx
                else:
                    # A = sl.block_diag(A, d_dx)
                    A = bmat([[A,None],[None,d_dx]], format='csr')
    return A

# @numba.jit(nopython=True)
def buildDerivYMatrix(grid_size):
    ny = grid_size[0]
    # logging.info('[INFO]>>> ny:{}'.format(ny))
    e = np.ones((ny - 1, 1))
    # logging.info('[INFO]>>> e.shape:{}'.format(e.shape))
    d_dy = sparse.spdiags(np.hstack((-e, e)).T, [0,1], ny - 1, ny)

    A = None

    for _ in xrange(0,grid_size[4]):
        for _ in xrange(0,grid_size[3]):
            for _ in xrange(0,grid_size[2]):
                for _ in xrange(0,grid_size[1]):
                    if(A is None):
                        A = d_dy
                    else:
                        A = bmat([[A,None],[None,d_dy]], format='csr')
    return A

def buildDerivZMatrix(grid_size):
    m = grid_size[0] * grid_size[1] * (grid_size[2] - 1)
    n = grid_size[0] * grid_size[1] * grid_size[2]
    e = np.ones((m, 1))
    d_dz = sparse.spdiags([-e, e], [0, grid_size[0] * grid_size[1]], m, n)	

    A = None

    for _ in xrange(0,grid_size[4]):
        for _ in xrange(0,grid_size[3]):
            if(A is None):
                A = d_dz
            else:
                # A = sl.block_diag(A, d_dz)
                A = bmat([[A,None],[None,d_dz]])
    return A

# @numba.jit(nopython=True)
def buildSecondDerivZMatrix(grid_size):
    m = grid_size[0] * grid_size[1] * (grid_size[2] - 2)
    n = grid_size[0] * grid_size[1] * grid_size[2]
    e = np.ones((m, 1))

    interior = sparse.spdiags(np.hstack((e, -2 * e, e)).T, \
        [0, grid_size[0] * grid_size[1], 2 * grid_size[0] * grid_size[1]], \
        m, n)

    boundary_z1 = makeBoundaryZ1(grid_size, n)
    boundary_zend = makeBoundaryZEnd(grid_size, n)
    logging.info('[INFO]>>> boundary_z1:{} interior:{}  boundary_zend:{}\n'.\
        format(boundary_z1.shape, interior.shape, boundary_zend.shape))
    cube = bmat([[boundary_z1], [interior], [boundary_zend]], format='csr')

    logging.info('[INFO]>>> cube:{}'.format(cube.shape))
    A = None

    for _ in xrange(0,grid_size[4]):
        for _ in xrange(0,grid_size[3]):
            if(A is None):
                # A = sl.block_diag(cube)
                A = cube
            else:
                # A = sl.block_diag(A, cube)
                A = bmat([[A,None],[None,cube]], format='csr')
    return A

def bguSlice(gamma,IF,EIF):
    ih = IF.shape[0]
    iw = IF.shape[1]
    gh = gamma.shape[0]
    gw = gamma.shape[1]
    gd = gamma.shape[2]
    ao = gamma.shape[3]
    ai = gamma.shape[4]
    x = repmat(np.arange(0,iw),ih,1)
    y = repmat(np.arange(0,ih).T,iw,1).T
    bg_coord_x = (x + 0.5) * (gw - 1) / iw
    bg_coord_y = (y + 0.5) * (gh - 1) / ih

    bg_coord_z = EIF * (gd - 1)

    bg_coord_xx = bg_coord_x + 1
    bg_coord_yy = bg_coord_y + 1
    bg_coord_zz = bg_coord_z + 1

    xx = np.linspace(0,gh,gh)
    yy = np.linspace(0,gw,gw)
    zz = np.linspace(0,gd,gd)

    affine_model  = get_td_list(ai, ao)
    for j in range(ai):
        for i in range(ao):
            my_inter_fun = rgi((xx,yy,zz),gamma[:,:,:,i,j],bounds_error=False)
            pts = np.array([bg_coord_xx,bg_coord_yy,bg_coord_zz]).T
            # print(np.max(gamma))
            affine_model[i][j] = my_inter_fun(pts).T
            print('[INFO]>>> affine_model[{}][{}].shape:{}'.format(i,j,affine_model[i][j].shape))
    ll = np.ones((ih, iw))
    ll_ = np.array([IF[:,:,0],IF[:,:,1],IF[:,:,2],ll])
    input1 = ll_.swapaxes(0,1).swapaxes(1,2)
    output = 0
    for i in range(ai):
        new_ = np.array([affine_model[0][i],affine_model[1][i],affine_model[2][i]])
        affine_model2 = new_.swapaxes(0,1).swapaxes(1,2)
        print('[INFO]>>> affine_model2.shape:{}'.format(affine_model2.shape))
        output = output + (affine_model2 * input1[:,:,i][:,:,None])
    return output	


def getDefaultAffineGridSize(input_image, output_image):
    input_height    = input_image.shape[0]
    input_width     = input_image.shape[1]
    input_channels  = input_image.shape[2]
    output_channels = output_image.shape[2]

    grid_size = np.rint([input_height / 16, input_width / 16, 8, \
        output_channels, input_channels + 1])

    return map(int,grid_size)