if [ $# -ne 1 ]; then
  echo "Usage: $0 folder"
  exit 3
fi

mkdir ../tasks/$1
if [ $? -eq 0 ] 
then
	cp -r * ../tasks/$1
	rm -rf ../tasks/$1/.git
fi
