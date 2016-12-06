
#!/bin/bash
set -e
set -x

python Assignment6.py -fold 5 data/a5a.train -test data/a5a.test > finalLog
