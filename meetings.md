# Notes Meetings
## 11.2023
Premier rendez-vous, explication du projet Terzina. Plus explication de l'existant du telescope CTA.

## 15.12.2023
Focus for next meeting : 
1. Count the "pe" outputs
2. From the sample inputs, at a given MHz given a bigger array of outputs (represents better time resolution)
3. Optional : Increase the output resolution better timing resolution

## 24.01.2024
Network should work the best on high NSB levels for long time operations.
Updating weights is possible, changing the architecture of the FPGA is not possible.

- Add metrics of how well the model is working
- Compute how much data reduction is possible to do based on simulated data

Sensor -> Threshold -> Datacube (8x8pixels x20samples)-> 1D CNN -> Is it a "pe event" / how many -> Tjark CNN -> Score "signalness" if we send to ground


## 20.03.2024
Fichier extremement proche

Waveform plus d'ASICS mais d'autres
CITIROC
Amplitude maximum seulement

Time over threshold up or down
20-30ns

30x30 5ns resolution

Pre filter -> trigger / event classification
Random / Gamma shower / Hadronic shower
Particle classification -> Gamma or Hadronic (most likely shower)

### Presentation
LA Lumière Cherenkov ou Cherenkov Radiation

Leonid pictures
IACT vs Terzina is different in how to detect

HAWC is a WCDs and not IACT
Recent :
    LST-1 add first of 4
    SST-1M add
Future :
    CTA
    ASTRI

Avoid interferences and change to bruit de fond

Measure any photo electron instead of cherenkov light

Photon multiplier to explain

Plot model better visualisation
Flops / Inputs / parameters

### Datacubes
y tables -> look file structure
https://github.com/cta-observatory/dl1-data-handler/tree/master

## 27.03.2024
Tjark CTLearn
-> Only use r0

LST_LST_LSTSiPMCam

## 17.05.2024

Recadrage du périmètre 

Step 1 : metrics et tests
Step 2 : Comparaison de CNN/RNN/KAN
Step 3 : Voir plus loin Stéréoscopie/CTLearn

Génération :
Sample rate fixe / impulse détecté doit matché avec les données datacubes

Métrique : 
Validation technique des vrai / faux positifs
Robustesse sur différents sample rate


LSTM intéressant ?
Kolmogorov-Arnold networks ?
Physics modeled driven networks


https://medium.com/@saadsalmanakram/kolmogorov-arnold-networks-a-comprehensive-guide-to-neural-network-advancement-5919fc8f81b1

https://arxiv.org/pdf/2405.08790


## 31.05.2024
Precision, recall et accuracy
minimum detection value of 1

# 12.07.2024
Lower amounts of energy showers more interesting in the scientific detection

Run Model Model in between R1 and CTLearn Trigger
< 5k Params should be good for ASICs 

ASIC chip 22nm

## TODO 
Execution time metrics
FLOPS per inference
Make Leonid data work
investigate Cycling learning rates

# 26.07.2024

LSTM stateful vs stateless
