#!/usr/bin/env bash
here=$(cd "$(dirname "$0")" && pwd)s
. $here/../.env/bin/activate
python training.py $1