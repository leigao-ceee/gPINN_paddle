#bash

for id in {1..10..1}
  do
    for name in 'gpinn'
      do
        for Nx in 5 10 15 20 25 30
          do
            python run_3.3.1.py --Nx_EQs ${Nx} --net_type ${name}+$"$id" --epochs_adam 50000
            echo ${name}+$"$id"+"-"+$"$Nx"+" completed!"
          done
      done
  done


