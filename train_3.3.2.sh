

 for name in 'pinn' 'gpinn'
  do
    for Nx in 10
      do
        python run_3.3.2.py --Nx_EQs ${Nx} --net_type ${name}
      done
      echo ${name} + "3.3.2 completed!"
  done