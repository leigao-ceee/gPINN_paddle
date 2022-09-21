for name in 'gpinn_rar' 'pinn_rar'
  do
    for i in {0..40..1}
      do
        python run_3.4.2.py --Nx_EQs 1500 --samp_ids ${i} --net_type ${name} --work_name "Burgers_2D"
      done
      echo ${name} + "3.4.2 completed!"
  done



#for name in 'pinn' 'gpinn'
#  do
#    for Nx in 1500 2000 2500 3000
#      do
#        python run_3.4.1.py --Nx_EQs ${Nx} --net_type ${name}
#      done
#      echo ${name} + "3.4.1 completed!"
#  done