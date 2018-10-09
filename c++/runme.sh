#!/bin/bash

set -e


#gcc -g -O2 -Wall -Wextra -Werror -std=c99 random.c -o random
gcc -g -O2 -Wall -Wextra -Werror -std=c99 capture.c -o capture

./capture
