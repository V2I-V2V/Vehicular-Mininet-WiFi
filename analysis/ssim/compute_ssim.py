#!/usr/bin/python
import matlab.engine
import time
import os, sys
from Queue import Queue
from threading import Thread

results = {}

class Worker(Thread):
    """ Thread executing tasks from a given tasks queue """

    def __init__(self, thread_i, tasks):
        Thread.__init__(self)
        self.tasks = tasks
        self.daemon = True
        self.thread_i = thread_i
        self.start()

    def run(self):
        while True:
            eng, frame_id, pcd_ref_name, pcd_dis_name = self.tasks.get()
            # print("Thread %d loading url... %s, queue size %d" % (self.thread_i, url, self.tasks.qsize()))
            try:
                ssim = eng.compute_ssim(pcd_ref_name, pcd_dis_name, nargout=1)
                # print(frameID, ref_image_path, dis_image_path, ssim)
                results[int(frame_id)] = ssim
            except Exception as e:
                # An exception happened in this thread
                print(e)
            finally:
                # Mark this task as done, whether an exception happened or not
                self.tasks.task_done()


class ComputePool:
    """ Pool of threads consuming tasks from a queue """

    def __init__(self, num_threads):
        self.tasks = Queue(num_threads)
        for thread_i in range(num_threads):
            Worker(thread_i, self.tasks)

    def add_task(self, task):
        """ Add a task to the queue """
        self.tasks.put(task)

    def map(self, tasks):
        """ Add a list of tasks to the queue """
        for task in tasks:
            self.add_task(task)

    def wait_completion(self):
        """ Wait for completion of all the tasks in the queue """
        self.tasks.join()


def parse_config_from_file(filename):
    config_params = {}
    config_file = open(filename, 'r')
    for line in config_file.readlines():
        if "=" in line:
            key = line.split("=")[0]
            value = line.split("=")[1][:-1]  # remove "\n"
            config_params[key] = value
    config_file.close()
    return config_params


def get_eng(frame_id, n, engs):
    return engs[frame_id % n]


def main():
    ref_dir = sys.argv[1]
    data_folder = sys.argv[2]
    dis_dir = data_folder + "/output/"
    nframes = int(sys.argv[3])
    if nframes == -1:
        nframes = 10000
    num_threads = int(sys.argv[4])
    configs = parse_config_from_file(data_folder + '/config.txt')
    num_of_nodes = int(configs["num_of_nodes"])
    frameinfo = []
    start = time.time()
    n_engs = 6
    engs = [matlab.engine.start_matlab() for x in range(n_engs)]
    now = time.time()
    print("time taken to start " + str(n_engs) + " matlab(s): " + str("{:.2f}".format(now - start)) + " sec")
    for frame_id in range(nframes):
        pcd_dis_name = dis_dir + '/merged_frame' + str(frame_id) + '.pcd'
        if os.path.exists(pcd_dis_name):
            pcd_ref_name = ref_dir + '/'+ str(frame_id % 80).zfill(6) + '_' + str(num_of_nodes) + '.pcd'
            frameinfo.append((get_eng(frame_id, n_engs, engs), frame_id, pcd_ref_name, pcd_dis_name))
    
    compute_pool = ComputePool(num_threads)
    for info in frameinfo:
        compute_pool.add_task(info)
    compute_pool.wait_completion()
    for frameID in sorted(results.keys()):
        print(frameID, results[frameID])
    now = time.time()
    print("time taken in total: " + str("{:.2f}".format(now - start)) + " sec")


if __name__ == "__main__":
    main()
