# -*- coding: utf-8 -*-
"""
###################################################################
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

copyright (c) 2020, Peter Szutor

@author: Peter Szutor, Hungary, szppaks@gmail.com
Created on Wed Feb 26 17:23:24 2020
###################################################################



Octree-based lossy point-cloud compression with open3d and numpy
Average compressing rate (depends on octreee depth setting parameter): 0.012 - 0.1   


Input formats: You can get a list of supported formats from : http://www.open3d.org/docs/release/tutorial/Basic/file_io.html#point-cloud
               (xyz,pts,ply,pcd)

Usage:
    
Dependencies: Open3D, Numpy  (You can install theese modules:  pip install open3d, pip install numpy)    
    
Compress a point cloud:
    
octreezip(<filename>,<depth>) -> <result>
<filename>: (str) Point Cloud file name. Saved file name: [filename without ext]_ocz.npz  (Yes, it's a numpy array file)
<depth>   : (str) Octree depth. You can try 11-16 for best result. Bigger depht results higher precision and bigger compressed file size.
<result>  : (str) If the compressing was success you get: "Compressed into:[comp.file name] | Storing resolution:0.003445". Storing resolution means the precision.
                  The PC file is missing or bad: "PC is empty, bad, or missing"
                  Other error: "Error: [error message]"
                  

Uncompressing:
octreeunzip(<filename>) -> <result>
<filename>: (str) Zipped Point Cloud file name (npz). Saved file name: [filename].xyz  (standard XYZ text file)
<result>  : (str) If the compressing was success you get: "Saved: [filename].xyz"
                  Other error: "Error: [error message]"
                  


"""
import numpy as np
import time
import blosc


def d1halfing_fast(pmin,pmax,pdepht):
    return np.linspace(pmin,pmax,2**int(pdepht)+1)

class OcTree:

    def __init__(self, ptcl, depth=10) -> None:
        self.points = ptcl
        self.depth = depth
        self.encoded = None
        self.sorted_index = None
        self.decoded = None

    @staticmethod
    def octreecodes(ppoints,pdepht):
        minx=np.amin(ppoints[:,0])
        maxx=np.amax(ppoints[:,0])
        miny=np.amin(ppoints[:,1])
        maxy=np.amax(ppoints[:,1])
        minz=np.amin(ppoints[:,2])
        maxz=np.amax(ppoints[:,2])
        xletra=d1halfing_fast(minx,maxx,pdepht)
        yletra=d1halfing_fast(miny,maxy,pdepht)
        zletra=d1halfing_fast(minz,maxz,pdepht)
        otcodex=np.searchsorted(xletra,ppoints[:,0],side='right')-1
        otcodey=np.searchsorted(yletra,ppoints[:,1],side='right')-1
        otcodez=np.searchsorted(zletra,ppoints[:,2],side='right')-1
        ki=otcodex*(2**(pdepht*2))+otcodey*(2**pdepht)+otcodez
        return (ki,minx,maxx,miny,maxy,minz,maxz)

    def encode(self):
        occ=OcTree.octreecodes(self.points, self.depth)
        self.sorted_index = np.argsort(occ[0])
        occsorted=occ[0][self.sorted_index]
        # print(len(occsorted.tostring()), occsorted.dtype)

        encode_buffer = blosc.compress(occsorted.tostring()[:6000], typesize=8)
        # print(len(occsorted.tostring())/6000)
        # print(len(encode_buffer))
        decode_buffer = blosc.decompress(encode_buffer)
        np_arr = np.fromstring(decode_buffer, dtype=np.int64)
        # print(np_arr.shape)
        # prec=np.amax(np.asarray([occ[2]-occ[1],occ[4]-occ[3],occ[6]-occ[5]])/(2**self.depth))
        paramarr=np.asarray([self.depth,occ[1],occ[2],occ[3],occ[4],occ[5],occ[6]]) #depth and boundary
        # print(len(paramarr.tostring()))
        # print(paramarr.dtype)
        self.encoded = (occsorted, paramarr)

    def decode(self):
        if self.encoded is not None:
            pcpoints=self.encoded[0]
            pcparams=self.encoded[1]
            pdepht=(pcparams[0])
            minx=(pcparams[1])
            maxx=(pcparams[2])
            miny=(pcparams[3])
            maxy=(pcparams[4])
            minz=(pcparams[5])
            maxz=(pcparams[6])
            xletra=d1halfing_fast(minx,maxx,pdepht)
            yletra=d1halfing_fast(miny,maxy,pdepht)
            zletra=d1halfing_fast(minz,maxz,pdepht)    
            occodex=(pcpoints/(2**(pdepht*2))).astype(int)
            occodey=((pcpoints-occodex*(2**(pdepht*2)))/(2**pdepht)).astype(int)
            occodez=(pcpoints-occodex*(2**(pdepht*2))-occodey*(2**pdepht)).astype(int)
            koorx=xletra[occodex]
            koory=yletra[occodey]
            koorz=zletra[occodez]
            points=np.array([koorx,koory,koorz]).T
            self.decoded=points.astype(np.float32)
            # print(self.decoded.shape)
    
    @staticmethod
    def loss_resillient_encode(points, depth=7):
        encoded_packets = []
        occ = OcTree.octreecodes(points, depth)
        sorted_index = np.argsort(occ[0])
        occsorted=occ[0][sorted_index]
        paramarr=np.asarray([depth,occ[1],occ[2],occ[3],occ[4],occ[5],occ[6]]) #depth and boundary
        metadata = paramarr.tostring()
        serialized = occsorted.tostring()
        for i in range(int(len(serialized) / 4000)):
            encoded_buffter = blosc.compress(serialized[i*4000:(i+1)*4000], typesize=8)
            packet_payload = metadata + encoded_buffter
            encoded_packets.append(packet_payload)
        
        if len(serialized) % 4000 != 0:
            encoded_buffter = blosc.compress(serialized[:-(len(serialized) % 4000)], typesize=8)
            packet_payload = metadata + encoded_buffter
            encoded_packets.append(packet_payload)
        return encoded_packets
        
    
    @staticmethod
    def decode_partial(encode_buffer):
        paramarr_str = encode_buffer[:56]
        pcparams = np.fromstring(paramarr_str, dtype=np.float64)
        decode_buffer = blosc.decompress(encode_buffer[56:])
        pcpoints = np.fromstring(decode_buffer, dtype=np.int64)
        pdepht=(pcparams[0])
        minx=(pcparams[1])
        maxx=(pcparams[2])
        miny=(pcparams[3])
        maxy=(pcparams[4])
        minz=(pcparams[5])
        maxz=(pcparams[6])
        xletra=d1halfing_fast(minx,maxx,pdepht)
        yletra=d1halfing_fast(miny,maxy,pdepht)
        zletra=d1halfing_fast(minz,maxz,pdepht)    
        occodex=(pcpoints/(2**(pdepht*2))).astype(int)
        occodey=((pcpoints-occodex*(2**(pdepht*2)))/(2**pdepht)).astype(int)
        occodez=(pcpoints-occodex*(2**(pdepht*2))-occodey*(2**pdepht)).astype(int)
        koorx=xletra[occodex]
        koory=yletra[occodey]
        koorz=zletra[occodez]
        points=np.array([koorx,koory,koorz]).T
        return points


def filter_regions(ptcl, limit=80):
    mask1 = ptcl[:, 0] >= -limit
    mask2 = ptcl[:, 0] <= limit
    mask3 = ptcl[:, 1] >= -limit
    mask4 = ptcl[:, 1] <= limit
    mask = np.logical_and(mask1, mask2)
    mask = np.logical_and(mask, mask3)
    mask = np.logical_and(mask, mask4)
    return ptcl[mask]



if __name__ == '__main__':
    ptcl = np.load('/home/mininet-wifi/dense_160_town05_far/lidar/202/800.npy')
    octree = OcTree(ptcl, 12)
    octree.encode()
    # octree.decode()
    start_t = time.time()
    rst = OcTree.loss_resillient_encode(ptcl, 7)
    print(time.time() - start_t)
    
    print(len(rst[0]))
    start_t = time.time()
    for packet in rst:
        OcTree.decode_partial(packet)
    print(time.time() - start_t)
        