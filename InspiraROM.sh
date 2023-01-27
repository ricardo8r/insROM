#!/bin/bash

#Put all arguments into the arguments array
declare -a arguments
count=0; 
for i in $*
do 
   arguments[$count]=$i;
   count=$((count+1));
done
narg=$count;

basedir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )";

executable="python3 $basedir/insROM.py";
runtype="";
nprocs=1;

case="";
problem="";
nbasis="50";
library="numpy";
sets="single";
typeb="rSVD";
method="POD";
optfil="1";

count=0;
while [ $count -lt $narg ];
do
   if [ $count -eq 0 ]; then
      case="${arguments[$count]}";
   elif [ ${arguments[$count]} = "-n" ]; then
      nprocs=${arguments[$((count+1))]}; 
      runtype="mpirun -n ${arguments[$((count+1))]}" 
   elif [ ${arguments[$count]} = "-p" ]; then
      problem="${arguments[$((count+1))]}" 
   elif [ ${arguments[$count]} = "-r" ]; then
      nbasis="${arguments[$((count+1))]}" 
   elif [ ${arguments[$count]} = "-l" ]; then
      library="${arguments[$((count+1))]}" 
   elif [ ${arguments[$count]} = "-m" ]; then
      method="${arguments[$((count+1))]}" 
   elif [ ${arguments[$count]} = "-t" ]; then
      typeb="${arguments[$((count+1))]}" 
   elif [ ${arguments[$count]} = "-s" ]; then
      sets="${arguments[$((count+1))]}" 
   elif [ ${arguments[$count]} = "-f" ]; then
      optfil="1" 
   elif [ ${arguments[$count]} = "-nf" ]; then
      optfil="0" 
   fi
   count=$((count+1));
done

options="Case: $case, Sets: $sets, Method: $method, Type: $typeb, Library: $library, Problem: $problem, #Basis: $nbasis, Filter: $optfil";
echo $options;

runstring="$runtype $executable -c $case -s $sets -m $method -t $typeb -l $library -p $problem -r $nbasis -f $optfil";
eval $runstring;
