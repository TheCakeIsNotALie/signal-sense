# How to produce data

Use the pyeventio_example project : https://github.com/burmist-git/pyeventio_example/tree/master

After compiling, modify the runana.sh file in the if clause `-EvPerEv` :
Modify the `inRootFile` `outputFile` to use the wanted simulation files and `eventID` for the specific event wanted to turn into a waveform.

Then run it `source runana.sh -EvPerEv` and keep the output file.