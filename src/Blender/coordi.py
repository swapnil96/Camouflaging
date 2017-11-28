import cPickle as pickle
import numpy as np

file = open('/tmp/data_pos.out', 'r')
coordinates = []
for line in file.readlines():
    t1, t2, t3 = map(float, line.split())
    temp = np.array([t1, t2, t3])
    coordinates.append(temp)

file.close()

file = open('/tmp/data_face.out', 'r')
face_data = []
for line in file.readlines():
    t1, t2, t3, t4 = map(int, line.split())
    temp = np.array([t1, t2, t3, t4], dtype=np.uint8)
    face_data.append(temp)

file.close()

data = {'faces': np.array(face_data),
        'world_pos': np.array(coordinates)}

with open("/home/swapnil/Project/Test Data/cube.mat", "w") as myFile:
    pickle.dump(data, myFile)
