if [ $# -ne 1 ]; then
  echo "Usage: $0 <screen_name>"
  exit 3
fi
source deactivate
screen -S $1 -dm bash -c 'source activate tf10;python -u trader_running.py| tee output.txt'
