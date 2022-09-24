#bash
aa=1
c="$aa"
echo $c

for id in {1..10..1}
  do
    for name in 'gpinn' 'pinn'
      do
        for Nx in 5 10 15 20 25 30
          do
            python run_3.3.1.py --Nx_EQs ${Nx} --net_type ${name}+$"$id"
            echo ${name}+$"$id"+"-"+$"$Nx"+" completed!"
          done
      done
  done


