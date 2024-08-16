#!/bin/sh

module purge
module load GCC/11.2.0  OpenMPI/4.1.1 ROOT/6.24.06

ev_start=0
ev_stop=100

inRootFiles="/srv/beegfs/scratch/shares/heller/Leonid/mono-lst-sipm-pmma-3ns-v1_triggerless/gamma/000"
rndseed="1312312"

for ((evID=ev_start; evID<=ev_stop; evID++))
do
    binFileOut="/home/users/p/perrinya/scratch/bin_data/gamma_ev"$evID"_out.bin"
    srun --time "00:00:30" /home/users/p/perrinya/pyeventio_example/runana 333 "$inRootFiles" "$evID" "$binFileOut" "$rndseed"
done
