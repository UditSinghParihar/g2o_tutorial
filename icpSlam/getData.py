import open3d as o3d
from sys import argv, exit
import numpy as np
from scipy.spatial.transform import Rotation as R
import copy
import os


np.random.seed(42)


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


def getCloud(cube, color):
	vertices = []

	for ele in cube:	
		sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
		sphere.paint_uniform_color(color)

		trans = np.identity(4)
		trans[0, 3] = ele[0]
		trans[1, 3] = ele[1]
		trans[2, 3] = ele[2]

		sphere.transform(trans)
		vertices.append(sphere)

	return vertices


def getFrames():
	# posei = ( x, y, z, thetaZ(deg) )

	# poses = [[-8, 8, 0, -60], [-10, 4, 0, -30], [-12, 0, 0, 0], [-10, -4, 0, 30], [-8, -8, 0, 60]]
	poses = [[-12, 0, 0, 0], [-10, -4, 0, 30], [-8, -8, 0, 60], [-4, -12, 0, 75], [0, -16, 0, 80]]

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

	for i, pose in enumerate(poses):
		cube = []

		T = np.identity(4)
		T[0, 3], T[1, 3], T[2, 3] = pose[0], pose[1], pose[2]
		T[0:3, 0:3] = R.from_euler('z', pose[3], degrees=True).as_dcm()

		for pt in np.hstack((points, np.ones((points.shape[0], 1)))):
			ptLocal = np.linalg.inv(T) @ pt.reshape(4, 1)

			cube.append(ptLocal.squeeze(1)[0:3])

		cubes[i] = np.asarray(cube)

	return cubes


def addNoiseCubes(cubes, noise=0.15):
	noisyCubes = np.zeros(cubes.shape)

	for i in range(cubes.shape[0]):
		noiseMat = np.random.normal(0, noise, cubes[i].size).reshape(cubes[i].shape)
		noisyCubes[i] = cubes[i] + noiseMat

	return noisyCubes


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
	
	cloud1 = getCloud(cubes[0], [0.9, 0.2, 0])
	cloud2 = getCloud(cubes[1], [0, 0.2, 0.9])
	cloud3 = getCloud(cubes[2], [0.2, 0.9, 0])
	cloud4 = getCloud(cubes[3], [0.5, 0, 0.95])
	cloud5 = getCloud(cubes[4], [0.9, 0.45, 0])

	T1_2 = trans[0]
	T2_3 = trans[1]
	T3_4 = trans[2]
	T4_5 = trans[3]

	cloud2 = [ele.transform(T1_2) for ele in cloud2]
	cloud3 = [ele.transform(T1_2 @ T2_3) for ele in cloud3]
	cloud4 = [ele.transform(T1_2 @ T2_3 @ T3_4) for ele in cloud4]
	cloud5 = [ele.transform(T1_2 @ T2_3 @ T3_4 @ T4_5) for ele in cloud5]

	geometries = cloud1 + cloud2 + cloud3 + cloud4 + cloud5

	o3d.visualization.draw_geometries(geometries)


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


def writeRobotPose(trans, g2o):
	# Tw_1: 1 wrt w

	start = [-12, 0, 0, 0]

	Tw_1 = np.identity(4)
	Tw_1[0, 3], Tw_1[1, 3], Tw_1[2, 3] = start[0], start[1], start[2]
	Tw_1[0:3, 0:3] = R.from_euler('z', start[3], degrees=True).as_dcm()

	T1_2, T2_3, T3_4, T4_5 = trans[0], trans[1], trans[2], trans[3]

	Tw_2 = Tw_1 @ T1_2
	Tw_3 = Tw_2 @ T2_3
	Tw_4 = Tw_3 @ T3_4
	Tw_5 = Tw_4 @ T4_5

	pose1 = [Tw_1[0, 3], Tw_1[1, 3], Tw_1[2, 3]] + list(R.from_dcm(Tw_1[0:3, 0:3]).as_quat())
	pose2 = [Tw_2[0, 3], Tw_2[1, 3], Tw_2[2, 3]] + list(R.from_dcm(Tw_2[0:3, 0:3]).as_quat())
	pose3 = [Tw_3[0, 3], Tw_3[1, 3], Tw_3[2, 3]] + list(R.from_dcm(Tw_3[0:3, 0:3]).as_quat())
	pose4 = [Tw_4[0, 3], Tw_4[1, 3], Tw_4[2, 3]] + list(R.from_dcm(Tw_4[0:3, 0:3]).as_quat())
	pose5 = [Tw_5[0, 3], Tw_5[1, 3], Tw_5[2, 3]] + list(R.from_dcm(Tw_5[0:3, 0:3]).as_quat())

	posesRobot = [pose1, pose2, pose3, pose4, pose5]

	sp = ' '

	for i, (x, y, z, qx, qy, qz, qw) in enumerate(posesRobot):
		line = "VERTEX_SE3:QUAT " + str(i+1) + sp + str(x) + sp + str(y) + sp + str(z) + sp + str(qx) + sp + str(qy) + sp + str(qz) + sp + str(qw) + '\n'
		g2o.write(line)

	
def writeOdom(trans, g2o):	
	sp = ' '
	info = '20 0 0 0 0 0 20 0 0 0 0 20 0 0 0 20 0 0 20 0 20'

	for i, T in enumerate(trans):
		dx, dy, dz = T[0, 3], T[1, 3], T[2, 3]

		qx, qy, qz, qw = list(R.from_dcm(T[0:3, 0:3]).as_quat())
		
		line = "EDGE_SE3:QUAT " + str(i+1) + sp + str(i+2) + sp + str(dx) + sp + str(dy) + sp + str(dz) + sp + str(qx) + sp + str(qy) + sp + str(qz) + sp + str(qw) + sp +  info + '\n'

		g2o.write(line)


def writeCubeVertices(cubes, g2o):
	cube1 = cubes[0]

	start = [-12, 0, 0, 0]

	Tw_1 = np.identity(4)
	Tw_1[0, 3], Tw_1[1, 3], Tw_1[2, 3] = start[0], start[1], start[2]
	Tw_1[0:3, 0:3] = R.from_euler('z', start[3], degrees=True).as_dcm()

	cube = []

	for pt in np.hstack((cube1, np.ones((cube1.shape[0], 1)))):
		ptWorld = Tw_1 @ pt.reshape(4, 1)

		cube.append(ptWorld.squeeze(1)[0:3])

	quat = "0 0 0 1\n"
	sp = ' '

	for i, (x, y, z) in enumerate(cube):
		line = "VERTEX_SE3:QUAT " + str(i+6) + sp + str(x) + sp + str(y) + sp + str(z) + sp + quat
		
		g2o.write(line)


def writeLandmarkEdge(cubes, g2o):
	quat = "0 0 0 1"
	sp = ' '
	info = '40 0 0 0 0 0 40 0 0 0 0 40 0 0 0 0.000001 0 0 0.000001 0 0.000001' 

	for i, cube in enumerate(cubes):
		for j, (x, y, z) in enumerate(cube):
			line  = "EDGE_SE3:QUAT " + str(i+1) + sp + str(j+6) + sp + str(x) + sp + str(y) + sp + str(z) + sp + quat + sp +  info + '\n'

			g2o.write(line)


def writeG2o(trans, cubes):
	g2o = open("noise.g2o", 'w')

	g2o.write('# Robot poses\n\n')

	writeRobotPose(trans, g2o)

	g2o.write("\n # Cube vertices\n\n")

	writeCubeVertices(cubes, g2o)

	g2o.write('\n# Odometry edges\n\n')

	writeOdom(trans, g2o)

	g2o.write('\n# Landmark edges\n\n')

	writeLandmarkEdge(cubes, g2o)

	g2o.write("\nFIX 1\n")

	g2o.close()


def optimize():
	cmd = "g2o -robustKernel Cauchy -robustKernelWidth 1 -o {} -i 50 {} > /dev/null 2>&1".format(
		"opt.g2o", "noise.g2o")
	os.system(cmd)


def readG2o(fileName):
	f = open(fileName, 'r')
	A = f.readlines()
	f.close()

	poses = []

	for line in A:
		if "VERTEX_SE3:QUAT" in line:
			(ver, ind, x, y, z, qx, qy, qz, qw, newline) = line.split(' ')

			if int(ind) <= 5:
				T = np.identity(4)
				T[0, 3], T[1, 3], T[2, 3] = x, y, z
				T[0:3, 0:3] = R.from_quat([qx, qy, qz, qw]).as_dcm()

				poses.append(T)

	poses = np.asarray(poses)

	return poses


def getRelativeEdge(poses):
	T1_2 = np.linalg.inv(poses[0]) @ poses[1]
	T2_3 = np.linalg.inv(poses[1]) @ poses[2]
	T3_4 = np.linalg.inv(poses[2]) @ poses[3]
	T4_5 = np.linalg.inv(poses[3]) @ poses[4]

	trans = np.array([T1_2, T2_3, T3_4, T4_5])

	return trans


if __name__ == '__main__':
	vertices, points = getVertices()
	frames, poses = getFrames()

	visualizeData(vertices, frames)

	gtCubes = getLocalCubes(points, poses)
	noisyCubesHigh = addNoiseCubes(gtCubes, noise=1.8)
	noisyCubesLow = addNoiseCubes(gtCubes, noise=0.15)

	trans = icpTransformations(noisyCubesHigh)

	registerCubes(trans, noisyCubesLow)

	writeG2o(trans, noisyCubesLow)

	optimize()

	optPoses = readG2o("opt.g2o")
	optEdges = getRelativeEdge(optPoses)

	registerCubes(optEdges, noisyCubesLow)