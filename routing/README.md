# Usage


## Add route
```
./simple-route add to <ip_addr> dev <interface> via <gw_addr>
```

## Del route

```
./simple-route del to <ip_addr> dev <interface> via <gw_addr>
```

## MISC

The example code uses linux Netlink socket to update routing table in the kernel. For more information, please check the [linux man page](https://man7.org/linux/man-pages/man7/rtnetlink.7.html).
