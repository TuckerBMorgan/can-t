# Coding principles
1. Programs that use this should feel close to pytorch python / it should feel trivial to copy over a pytorch program to this code base
2. in places where python has default arugments (keepdims for example) force every callsite to call it out, don't follow a pattern of 
    sum(dimensions), sum_with_keep_alive(dimensions, keep_alive)
