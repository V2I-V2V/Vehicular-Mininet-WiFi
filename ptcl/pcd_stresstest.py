import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pointcloud
import ptcl_utils
import getpass

if __name__ == '__main__':
    usrname = getpass.getuser()
    ptcl = ptcl_utils.read_ptcl_data('/home/' + usrname + '/V2I+V2V/carla-bin/lidar/86/velodyne/1000.bin')
    # ptcl = ptcl_utils.read_ptcl_data('000001.bin')
    cl, qb = 10, 7
    encoded, ratio = pointcloud.dracoEncode(ptcl, cl, qb)
    decoded = pointcloud.dracoDecode(encoded)
    print(1 - ratio)
    print(2*ratio)
    print(ptcl.shape[0], decoded.shape[0])
    ptcl_utils.draw_3d(decoded)
    ptcl_utils.save_ptcl(decoded, './1000-%d-%d.bin' % (cl, qb))
