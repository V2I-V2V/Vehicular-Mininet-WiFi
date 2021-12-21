shced_to_displayed_name = {
    'combined-adapt': 'Harbor', 'v2v' : 'V2V', 'v2i-adapt': 'V2I-adapt', 'v2v-adapt': 'V2V-adapt', 'v2i': 'V2I',
    'combined-no-fallback-adapt': 'Harbor (w/o fallback)', 
    'combined-no-group-adapt': 'Harbor (w/o grouping)',
    'combined-deadline-unaware-adapt': 'Prioritization',
    'combined-no-prioritization-adapt': 'DDL-aware',
    'combined-unoptimized-delivery-adapt': 'Baseline',
    'random-adapt': 'Random',
    'distributed-adapt': 'Distributed',
    'minDist-adapt': 'minDist',
    'bwAware-adapt': 'V2I-BW',
    'routeAware-adapt': 'V2V-intf'
    
}
sched_to_color = {'minDist-adapt': 'r', 'random-adapt': 'limegreen', 'distributed-adapt': 'purple', 'combined': 'g',\
    'combined-adapt': 'midnightblue', 'bwAware-adapt': 'chocolate', 'combined-op_min-min': 'blueviolet',
    'combined-loc': 'brown', 'combined-op_sum-min': 'darkorange',  'routeAware-adapt': 'fuchsia',
    'combined-op_sum-harmonic': 'cyan', 'v2i': 'orange', 'combined-deadline': 'olive',
    'v2v' : 'crimson', 'v2i-adapt': 'forestgreen', 'v2v-adapt': 'darkviolet',
    'combined-no-fallback-adapt': 'maroon', 'combined-deadline-unaware-adapt': 'g',
    'combined-no-prioritization-adapt': 'brown',
    'combined-unoptimized-delivery-adapt': 'r',
    'combined-no-group-adapt': 'blueviolet'}

sched_to_marker = {'combined-adapt': 's', 'v2v' : '^', 
                   'v2i-adapt': 'h', 'v2v-adapt': 'X', 'v2i': 'o',
                    'combined-no-fallback-adapt': '^', 
                    'combined-deadline-unaware-adapt': '^',
                    'combined-no-prioritization-adapt': 'h',
                    'combined-unoptimized-delivery-adapt': 'X',
                    'random-adapt': '^',
                    'distributed-adapt': 'h',
                    'minDist-adapt': 'X',
                    'bwAware-adapt': 'o',
                    'routeAware-adapt': '^',
                    'combined-no-group-adapt': '^'}