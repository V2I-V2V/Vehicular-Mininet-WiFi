import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pointcloud
import ptcl_utils
import getpass


if __name__ == '__main__':
    usrname = getpass.getuser()
    ptcl = ptcl_utils.read_ptcl_data('/home/'+usrname+'/carla-bin/86/velodyne/1000.bin')
    encoded, ratio = pointcloud.dracoEncode(ptcl, 10, 10)
    decoded = pointcloud.dracoDecode(encoded)
    print(ratio)
    print(ptcl.shape[0], decoded.shape[0])
    ptcl_utils.draw_3d(decoded)
