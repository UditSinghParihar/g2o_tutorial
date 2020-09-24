import open3d as o3d
from sys import argv, exit
import numpy as np
from scipy.spatial.transform import Rotation as R
import copy


def getVertices():
	points = [[0, 8, 8], [0, 0, 8], [0, 0, 0], [0, 8, 0], [8, 8, 8], [8, 0, 8], [8, 0, 0], [8, 8, 0]]

	vertices = []

	for ele in points:
		if(ele is not None):
			sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
			sphere.paint_uniform_color([0.9, 0.2, 0])

			trans = np.identity(4)
			trans[0, 3] = ele[0]
			trans[1, 3] = ele[1]
			trans[2, 3] = ele[2]

			sphere.transform(trans)
			vertices.append(sphere)

	return vertices, points


def getFrames():
	# posei = ( x, y, z, thetaZ(deg) )

	# poses = [[-4, 4, 0, -60], [-5, 2, 0, -30], [-6, 0, 0, 0], [-5, -2, 0, 30], [-4, -4, 0, 60]]
	poses = [[-8, 8, 0, -60], [-10, 4, 0, -30], [-12, 0, 0, 0], [-10, -4, 0, 30], [-8, -8, 0, 60]]

	frames = []

	for pose in poses:
		T = np.identity(4)
		T[0, 3], T[1, 3], T[2, 3] = pose[0], pose[1], pose[2]
		T[0:3, 0:3] = R.from_euler('z', pose[3], degrees=True).as_dcm()

		frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.2, origin=[0, 0, 0])
		frame.transform(T)
		frames.append(frame)

	return frames, poses


def visualizeData(vertices, frames):
	geometries = []
	geometries = geometries + vertices + frames

	o3d.visualization.draw_geometries(geometries)


def getLocalCubes(points, poses):
	# Returns local point cloud cubes
	
	points = np.array(points)
	poses = np.array(poses)
	
	nPoses, nPoints, pointDim = poses.shape[0], points.shape[0], points.shape[1]
	cubes = np.zeros((nPoses, nPoints, pointDim))

	for i in range(nPoses):
		cube = points - poses[i, 0:3]

		# noise = np.random.normal(0, 0.3, cube.size).reshape(cube.shape)

		# cubes[i] = cube + noise
		cubes[i] = cube

	return cubes


def addNoiseCubes(cubes, noise=0.15):
	noisyCubes = np.zeros(cubes.shape)

	for i in range(cubes.shape[0]):
		noiseMat = np.random.normal(0, noise, cubes[i].size).reshape(cubes[i].shape)
		noisyCubes[i] = cubes[i] + noiseMat

	return noisyCubes


def addNoiseTrans(trans, noise=0.3):
	noisyTrans = np.zeros(trans.shape)

	for i in range(trans.shape[0]):
		noiseMat = np.random.normal(0, noise, trans[i].size).reshape(trans[i].shape)
		noisyTrans[i] = trans[i] + noiseMat

	return noisyTrans


def draw_registration_result(source, target, transformation):
	source_temp = copy.deepcopy(source)
	target_temp = copy.deepcopy(target)
	source_temp.paint_uniform_color([1, 0.706, 0])
	target_temp.paint_uniform_color([0, 0.651, 0.929])
	source_temp.transform(transformation)

	vis = o3d.visualization.Visualizer()
	vis.create_window()
	vis.add_geometry(source_temp)
	vis.add_geometry(target_temp)
	vis.get_render_option().point_size = 15
	vis.run()
	vis.destroy_window()


def registerCubes(trans, cubes):
	# Registering noisy cubes in first frame
	
	print(trans.shape, cubes.shape)

	
def icpTransformations(cubes):
	# T1_2 : 2 wrt 1 

	P1 = cubes[0]
	P2 = cubes[1]
	P3 = cubes[2]
	P4 = cubes[3]
	P5 = cubes[4]

	pcd1, pcd2, pcd3, pcd4, pcd5 = (o3d.geometry.PointCloud(), o3d.geometry.PointCloud(), 
	o3d.geometry.PointCloud(), o3d.geometry.PointCloud(), o3d.geometry.PointCloud())

	pcd1.points = o3d.utility.Vector3dVector(P1)
	pcd2.points = o3d.utility.Vector3dVector(P2)
	pcd3.points = o3d.utility.Vector3dVector(P3)
	pcd4.points = o3d.utility.Vector3dVector(P4)
	pcd5.points = o3d.utility.Vector3dVector(P5)

	corr = np.array([(i, i) for i in range(8)]) 

	p2p = o3d.registration.TransformationEstimationPointToPoint()

	T1_2 = p2p.compute_transformation(pcd2, pcd1, o3d.utility.Vector2iVector(corr))
	T2_3 = p2p.compute_transformation(pcd3, pcd2, o3d.utility.Vector2iVector(corr))
	T3_4 = p2p.compute_transformation(pcd4, pcd3, o3d.utility.Vector2iVector(corr))
	T4_5 = p2p.compute_transformation(pcd5, pcd4, o3d.utility.Vector2iVector(corr))

	# draw_registration_result(pcd2, pcd1, T1_2)

	trans = np.array([T1_2, T2_3, T3_4, T4_5])

	return trans


def writeG2o(trans, cubes):
	pass


if __name__ == '__main__':
	vertices, points = getVertices()
	frames, poses = getFrames()

	# visualizeData(vertices, frames)

	gtCubes = getLocalCubes(points, poses)
	noisyCubes = addNoiseCubes(gtCubes)

	gtTrans = icpTransformations(gtCubes)
	noisyTrans = addNoiseTrans(gtTrans)

	registerCubes(noisyTrans, noisyCubes)

	# writeG2o(trans, cubes)