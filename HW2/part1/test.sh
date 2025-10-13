#! /bin/bash

command="./pi.out 4 100000000"

# run command until its 5th output character is not 1

while true
do
    output=$($command)
    echo "Output: $output"
    first_char=${output:0:1}
    fifth_char=${output:4:1}
    if [ "$fifth_char" != "1" ]; then
        echo "Test failed."
        exit 1
    fi
done