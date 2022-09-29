 for id in {1..10..1}
  do
    for name in 'pinn'
      do
        for Nx in 5 10 15 20 25 30
          do
            python run_3.3.1.py --Nx_EQs ${Nx} --net_type ${name}+$"$id" --epochs_adam 50000
            echo ${name}+$"$id"+"-"+$"$Nx"+" completed!"
          done
      done
  done


 for name in 'gpinn' 'pinn'
  do
    for Nx in 5 10 15 20 25 30
      do
        python run_3.3.2.py --Nx_EQs ${Nx} --net_type ${name}
      done
      echo ${name} + "3.3.2 completed!"
  done