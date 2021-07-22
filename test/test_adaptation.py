import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_latency(e2e_frame_latency):
        recent_latency = 0
        recent_latencies = sorted(e2e_frame_latency.items(), key=lambda item: -item[0])
        cnt = 0
        for id, latency in recent_latencies:
            cnt += 1
            recent_latency += latency
            # print(id, latency)
            if cnt == 10:
                break
        recent_latency /= cnt
        return recent_latency

if __name__ == "__main__":
    
    print(get_latency(1, {1:1, 2:2}))