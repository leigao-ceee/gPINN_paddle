
for id in {1..10..1}
  do
    for Nx in 25 30 40 50 60 80 100 120 150
      do
        python run_3.2.2.py --Nx_EQs ${Nx}  --net_type "pinn"+$"$id" --epochs_adam 20000
        echo ${name}+$"$g"+"-"+$"$Nx"+" completed!"
      done
  done
#

for id in {1..10..1}
  do
    for Nx in 25 30 40 50 60 80 100 120 150
      do
          for g in  0.01 0.1 1.0
            do
              python run_3.2.2.py --Nx_EQs ${Nx} --g_weight ${g}  --net_type "gpinn"+$"$g"+$"$id" --epochs_adam 20000
              echo ${name}+$"$g"+"-"+$"$Nx"+" completed!"
            done
      done
  done





