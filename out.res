SDPA start at [Sat Aug  8 18:51:04 2026]
param  is param.sdpa 
data   is prg.dat  : dense
out    is out.res
NumThreads  is set as 1
Schur computation : DENSE 
   mu      thetaP  thetaD  objP      objD      alphaP  alphaD  beta 
 0 1.0e+04 1.0e+00 1.0e+00 -0.00e+00 +1.00e+02 1.0e+00 9.1e-01 2.00e-01
 1 1.4e+03 0.0e+00 9.1e-02 +1.39e+02 +1.00e+01 1.0e+00 1.0e+00 2.00e-01
 2 1.6e+02 0.0e+00 1.8e-17 +1.65e+02 +1.00e+00 1.0e+00 1.0e+00 1.00e-01
 3 1.6e+01 3.5e-17 0.0e+00 +1.74e+01 +1.00e+00 1.0e+00 9.0e+01 1.00e-01
 4 1.6e+00 3.5e-17 0.0e+00 +2.64e+00 +1.00e+00 1.0e+00 1.0e+00 1.00e-01
 5 1.6e-01 3.5e-17 1.1e-18 +1.16e+00 +1.00e+00 1.0e+00 9.0e+01 1.00e-01
 6 1.6e-02 3.6e-17 9.9e-17 +1.02e+00 +1.00e+00 1.0e+00 1.0e+00 1.00e-01
 7 1.6e-03 3.6e-17 1.1e-18 +1.00e+00 +1.00e+00 1.0e+00 9.0e+01 1.00e-01
 8 1.6e-04 3.5e-17 9.9e-17 +1.00e+00 +1.00e+00 1.0e+00 1.0e+00 1.00e-01
 9 1.6e-05 3.5e-17 1.1e-18 +1.00e+00 +1.00e+00 1.0e+00 9.0e+01 1.00e-01
10 1.6e-06 3.6e-17 9.9e-17 +1.00e+00 +1.00e+00 1.0e+00 1.0e+00 1.00e-01
11 1.6e-07 3.5e-17 0.0e+00 +1.00e+00 +1.00e+00 9.0e-01 9.0e-01 1.00e-01

phase.value  = pdFEAS    
   Iteration = 11
          mu = +1.6399389999999993e-07
relative gap = +1.6399388301560182e-07
        gap  = +1.6399389646259976e-07
     digits  = +6.7851723508676347e+00
objValPrimal = +1.0000001639938965e+00
objValDual   = +1.0000000000000000e+00
p.feas.error = +3.5374001701470904e-15
d.feas.error = +0.0000000000000000e+00
total time   = 0.000106
** Parameters **
maxIteration = 40
epsilonStar  = +9.9999999999999995e-08
lambdaStar   = +1.0000000000000000e+02
omegaStar    = +2.0000000000000000e+00
lowerBound   = -1.0000000000000000e+05
upperBound   = +1.0000000000000000e+05
betaStar     = +1.0000000000000001e-01
betaBar      = +2.0000000000000001e-01
gammaStar    = +9.0000000000000002e-01
epsilonDash  = +9.9999999999999995e-08
xPrint       = %+8.3e 
XPrint       = %+8.3e 
YPrint       = %+8.3e 
infPrint     = %+10.16e 

                         Time(sec)  Ratio(% : MainLoop) 
 Predictor time  =       0.000018,  20.000000
 Corrector time  =       0.000004,  4.444444
 Make bMat time  =       0.000002,  2.222222
 Make bDia time  =       0.000000,  0.000000
 Make bF1  time  =       0.000000,  0.000000
 Make bF2  time  =       0.000000,  0.000000
 Make bF3  time  =       0.000000,  0.000000
 Make bPRE time  =       0.000000,  0.000000
 Make rMat time  =       0.000000,  0.000000
 Make gVec Mul   =       0.000001,  1.111111
 Make gVec time  =       0.000003,  3.333333
 Cholesky bMat   =       0.000002,  2.222222
 Ste Pre time    =       0.000000,  0.000000
 Ste Cor time    =       0.000003,  3.333333
 solve           =       0.000010,  11.111111
 sumDz           =       0.000001,  1.111111
 makedX          =       0.000002,  2.222222
 symmetriseDx    =       0.000000,  0.000000
 makedXdZ        =       0.000004,  4.444444
 xMatTime        =       0.000000,  0.000000
 zMatTime        =       0.000000,  0.000000
 invzMatTime     =       0.000000,  0.000000
 xMatzMatTime    =       0.000000,  0.000000
 EigxMatTime     =       0.000001,  1.111111
 EigzMatTime     =       0.000000,  0.000000
 EigxMatzMatTime =       0.000000,  0.000000
 updateRes       =       0.000001,  1.111111
 EigTime         =       0.000001,  1.111111
 sub_total_bMat  =       0.000088,  97.777778
 Main Loop       =       0.000090,  100.000000
 File Check      =       0.000000,  0.000000
 File Change     =       0.000000,  0.000000
 File Read       =       0.000016,  17.777778
 Total           =       0.000106,  117.777778

xVec = 
{+1.000e+00}
xMat = 
{
{+1.640e-07}
}
yMat = 
{
{+1.000e+00}
}
    main loop time = 0.000090
        total time = 0.000106
  file  check time = 0.000000
  file change time = 0.000000
  file   read time = 0.000016
SDPA end at [Sat Aug  8 18:51:04 2026]
ALL TIME = 0.000366
