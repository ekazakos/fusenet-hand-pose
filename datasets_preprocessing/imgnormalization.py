import math
import numpy as np
from scipy.ndimage.measurements import center_of_mass
import cv2


class ImgNormalization(object):
	"""Preprocess hand images to center hand and normalize"""
	def __init__(self, fx, fy, px, py, cube_size):

		self.fx = fx
		self.fy = fy
		self.px = px
		self.py = py
		self.cube_size = cube_size


	#TODO Move the following 3 functions to a different class if necessary 		
	def depth_to_uvd(self, depth):

		uvd = np.zeros((3, depth.shape[0], depth.shape[1]))
		uv = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]
		uvd[0] = uv[1]
		uvd[1] = uv[0]
		uvd[2] = depth
		return uvd

	def uvd_to_xyz(self, uvd):

		xyz = np.zeros(uvd.shape)
		xyz[0] = (uvd[0] - self.px)*uvd[2]/self.fx
		xyz[1] = (self.py - uvd[1])*uvd[2]/self.fy
		xyz[2] = uvd[2]

		return xyz

	def xyz_to_uvd(self, xyz):

		uvd = np.zeros(xyz.shape)
		uvd[0] = xyz[0]*self.fx/xyz[2] + self.px
		uvd[1] = self.py - xyz[1]*self.fy/xyz[2]
		uvd[2] = xyz[2]

		return uvd

	@staticmethod
	def calculate_com(depth_hand):
		"""
		Calculate the center of mass
		:param dpt: depth image
		:return: (x,y,z) center of mass
		"""

		dc = depth_hand.copy()
		cc = center_of_mass(dc > 0)
		num = np.count_nonzero(dc)
		com = np.array((cc[1]*num, cc[0]*num, dc.sum()), np.float)

		if num == 0:
		    return np.array((0, 0, 0), np.float)
		else:
		    return com/num

	@staticmethod
	def transform_point_2D(pt, M):
		"""
		Transform point in 2D coordinates
		:param pt: point coordinates
		:param M: transformation matrix
		:return: transformed point
		"""
		pt2 = np.asmatrix(M.reshape((3, 3))) * np.matrix([pt[0], pt[1], 1]).T
		return np.array([pt2[0] / pt2[2], pt2[1] / pt2[2]])

	def ptcl_normalization(self, depth, com3D):
		"""
		Center point cloud to 0 and normalize it to [-1, 1]

		Keyword arguments: 
		depth -- depth image (the initial before cropping)
		com3D -- center of mass in 3D
		cube_size -- size of the cube that used to crop hand area (default 250)

		Return:
		ptcl_normalized -- point cloud centered to 0 and normalized to [-1, 1]
		"""

		pcl_uvd = self.depth_to_uvd(depth)
		pcl_xyz = self.uvd_to_xyz(pcl_uvd)
		indr,indc = np.nonzero(pcl_xyz[2])

		ptcl_normalized = np.vstack((pcl_xyz[0,indr,indc],pcl_xyz[1,indr,indc],pcl_xyz[2,indr,indc]))

		ptcl_normalized[0]-=com3D[0]
		ptcl_normalized[1]-=com3D[1]
		ptcl_normalized[2]-=com3D[2]
		ptcl_normalized /= self.cube_size / 2

		return ptcl_normalized

	def joints3D_depth_normalization(self, joints3D, depth, com3D):
		"""
        Center depth and joints in 3D to 0 and normalize it to [-1, 1].
        
        Keyword arguments:
        joints3D -- joints in 3D
        com3D -- center of mass in 3D
		depth -- depth image that has been croped and scaled

        Return:
        joints3D_normalized -- joints in 3D centered to 0 and normalized to [-1, 1]
        depth_normalized -- depth centered to 0 and normalized to [-1, 1]
        """

		joints3D_normalized = np.clip((joints3D - com3D[:,None]) / (self.cube_size / 2), -1, 1)
		depth[depth == 0.] = com3D[2] + self.cube_size / 2.
		depth -= com3D[2]
		depth_normalized = depth / (self.cube_size / 2)

		return joints3D_normalized, depth_normalized

	# def getNDValue(self):
	# 	"""
 #        Get value of not defined depth value distances
 #        :return:value of not defined depth value
 #        """
	# 	if self.depth[self.depth < self.minDepth].shape[0] > self.depth[self.depth > self.maxDepth].shape[0]:
	# 		return stats.mode(self.depth[self.depth < self.minDepth])[0][0]
	# 	else:
	# 		return stats.mode(self.depth[self.depth > self.maxDepth])[0][0]

	def crop_scale_depth(self, depth, com, dsize=(128, 128)):
		"""
		Crops depth image using 3D bounding box centered at the CoM of hand
		and then resize it to a 128x128 image
		:param depth: depth image
		:param com: center of mass of hand
		:param size: size of 3D bounding box
		:param dsize: size of the scaled image
		:return: cropped and resized image and transformation matrix for joints 

		"""
		maxDepth = min(1500, depth.max())
		minDepth = max(10, depth.min())
        # set values out of range to 0
		depth[depth > maxDepth] = 0.
		depth[depth < minDepth] = 0.
		
		# calculate boundaries
		zstart = com[2] - self.cube_size / 2.
		zend = com[2] + self.cube_size / 2.
		xstart = int(math.floor((com[0] * com[2] / self.fx - self.cube_size / 2.) / com[2]*self.fx))
		xend = int(math.floor((com[0] * com[2] / self.fx + self.cube_size / 2.) / com[2]*self.fx))
		ystart = int(math.floor((com[1] * com[2] / self.fy - self.cube_size / 2.) / com[2]*self.fy))
		yend = int(math.floor((com[1] * com[2] / self.fy + self.cube_size / 2.) / com[2]*self.fy))

		# crop patch from source
		cropped = depth[max(ystart, 0):min(yend, depth.shape[0]), max(xstart, 0):min(xend, depth.shape[1])].copy()
		# add pixels that are out of the image in order to keep aspect ratio
		cropped = np.pad(cropped, ((abs(ystart)-max(ystart, 0), abs(yend)-min(yend, depth.shape[0])), 
		(abs(xstart)-max(xstart, 0),abs(xend)-min(xend, depth.shape[1]))), mode='constant', constant_values=0)
		msk1 = np.bitwise_and(cropped < zstart, cropped != 0)
		msk2 = np.bitwise_and(cropped > zend, cropped != 0)
		cropped[msk1] = zstart
		cropped[msk2] = 0.

		wb = (xend - xstart)
		hb = (yend - ystart)

		trans = np.asmatrix(np.eye(3, dtype=float))
		trans[0, 2] = -xstart
		trans[1, 2] = -ystart

		if wb > hb:
			sz = (dsize[0], hb * dsize[0] / wb)
		else:
			sz = (wb * dsize[1] / hb, dsize[1])

		roi = cropped

		if roi.shape[0] > roi.shape[1]:
 			scale = np.asmatrix(np.eye(3, dtype=float) * sz[1] / float(roi.shape[0]))
		else:
 			scale = np.asmatrix(np.eye(3, dtype=float) * sz[0] / float(roi.shape[1]))
		scale[2, 2] = 1

		rz = cv2.resize(roi, sz, interpolation=cv2.INTER_NEAREST)

		ret = np.ones(dsize, np.float) * zend  # use background as filler
		xstart = int(math.floor(dsize[0] / 2 - rz.shape[1] / 2))
		xend = int(xstart + rz.shape[1])
		ystart = int(math.floor(dsize[1] / 2 - rz.shape[0] / 2))
		yend = int(ystart + rz.shape[0])
		ret[ystart:yend, xstart:xend] = rz

		off = np.asmatrix(np.eye(3, dtype=float))
		off[0, 2] = xstart
		off[1, 2] = ystart

		return ret, off * scale * trans

	def crop_scale_rgb(self, rgb, depth, com, dsize=(128, 128, 3)):
		"""
		Crops depth image using 3D bounding box centered at the CoM of hand
		and then resize it to a 128x128 image
		:param depth: depth image
		:param com: center of mass of hand
		:param size: size of 3D bounding box
		:param dsize: size of the scaled image
		:return: cropped and resized image and transformation matrix for joints 

		"""

		# calculate boundaries
		xstart = int(math.floor((com[0] * com[2] / self.fx - self.cube_size / 2.) / com[2]*self.fx))
		xend = int(math.floor((com[0] * com[2] / self.fx + self.cube_size / 2.) / com[2]*self.fx))
		ystart = int(math.floor((com[1] * com[2] / self.fy - self.cube_size / 2.) / com[2]*self.fy))
		yend = int(math.floor((com[1] * com[2] / self.fy + self.cube_size / 2.) / com[2]*self.fy))

		# crop patch from source
		cropped = rgb[max(ystart, 0):min(yend, rgb.shape[0]), max(xstart, 0):min(xend, rgb.shape[1])].copy()

		# add pixels that are out of the image in order to keep aspect ratio
		cropped = np.pad(cropped, ((abs(ystart)-max(ystart, 0), abs(yend)-min(yend, rgb.shape[0])), 
		(abs(xstart)-max(xstart, 0),abs(xend)-min(xend, rgb.shape[1])), (0,0)), mode='constant', constant_values=0)


		wb = (xend - xstart)
		hb = (yend - ystart)

		if wb > hb:
			sz = (dsize[0], hb * dsize[0] / wb)
		else:
			sz = (wb * dsize[1] / hb, dsize[1])

		roi = cropped
		rz = cv2.resize(roi, sz)

		ret = np.zeros(dsize, np.uint8) 
		xstart = int(math.floor(dsize[0] / 2 - rz.shape[1] / 2))
		xend = int(xstart + rz.shape[1])
		ystart = int(math.floor(dsize[1] / 2 - rz.shape[0] / 2))
		yend = int(ystart + rz.shape[0])
		ret[ystart:yend, xstart:xend, :] = rz
		msk = np.bitwise_not(np.bitwise_or(depth==1., depth==-1.))
		return ret, msk