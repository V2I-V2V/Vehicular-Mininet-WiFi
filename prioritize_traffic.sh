interface=$1
tc qdisc add dev ${interface} root handle 1: prio 
tc filter add dev ${interface} protocol ip parent 1: prio 1 u32 match ip dport 8000 0xffff flowid 1:1
tc filter add dev ${interface} protocol ip parent 1: prio 1 u32 match ip dport 8888 0xffff flowid 1:1