#! /bin/sh
set -e

[ -p "c2py" ] || mkfifo c2py
[ -p "py2c" ] || mkfifo py2c

./test.py &
#./pretendC.py
#valgrind fuego << EOF
fuego << EOF
play b c2
genmove w
genmove b
genmove w
showboard
genmove b
showboard
EOF

