! Clean import set for the SCC k-caching kernel: full PARKIND1 (all kinds, no I/O) +
! I/O-stripped YOMCST/YOETHF/YOECLDP (declarations + TECLDP type, FILE_IO_MOD/loaders removed;
! their module variables become SDFG free symbols -> config-propagated) + YOMPHYDER stub.

! (C) Copyright 1988- ECMWF.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
!
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation
! nor does it submit to any jurisdiction.

MODULE PARKIND1
!
!     *** Define usual kinds for strong typing ***
!
IMPLICIT NONE
SAVE
!
!     Integer Kinds
!     -------------
!
INTEGER, PARAMETER :: JPIT = SELECTED_INT_KIND(2)
INTEGER, PARAMETER :: JPIS = SELECTED_INT_KIND(4)
INTEGER, PARAMETER :: JPIM = SELECTED_INT_KIND(9)
INTEGER, PARAMETER :: JPIB = SELECTED_INT_KIND(12)

!Special integer type to be used for sensative adress calculations
!should be *8 for a machine with 8byte adressing for optimum performance
#ifdef ADDRESS64
INTEGER, PARAMETER :: JPIA = JPIB
#else
INTEGER, PARAMETER :: JPIA = JPIM
#endif

!
!     Real Kinds
!     ----------
!
INTEGER, PARAMETER :: JPRT = SELECTED_REAL_KIND(2,1)
INTEGER, PARAMETER :: JPRS = SELECTED_REAL_KIND(4,2)
INTEGER, PARAMETER :: JPRM = SELECTED_REAL_KIND(6,37)
#ifdef HALF
INTEGER, PARAMETER :: JPRB = 2  ! IEEE FP16 half precision (NVHPC real(2))
#  ifdef HALF_RESTRICTED
INTEGER, PARAMETER :: JPRL = SELECTED_REAL_KIND(6,37)  ! SC2026: FP32 for values that overflow FP16 (>65504)
#  else
INTEGER, PARAMETER :: JPRL = JPRB                       ! SC2026: aggressive — everything FP16
#  endif
#elif defined(SINGLE)
INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(6,37)
INTEGER, PARAMETER :: JPRL = JPRB
#else
INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13,300)
INTEGER, PARAMETER :: JPRL = JPRB
#endif

! Double real for C code and special places requiring
!    higher precision.
INTEGER, PARAMETER :: JPRD = SELECTED_REAL_KIND(13,300)


! Logical Kinds for RTTOV....

INTEGER, PARAMETER :: JPLM = JPIM   !Standard logical type

END MODULE PARKIND1

! (C) Copyright 1988- ECMWF.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
!
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation
! nor does it submit to any jurisdiction.

MODULE YOMCST

USE PARKIND1,    ONLY : JPRB, JPRL


IMPLICIT NONE

SAVE

!     ------------------------------------------------------------------

!*    Common of physical constants
!     You will find the meanings in the annex 1 of the documentation

! A1.0 Fundamental constants
! * RPI          : number Pi
! * RCLUM        : light velocity
! * RHPLA        : Planck constant
! * RKBOL        : Bolzmann constant
! * RNAVO        : Avogadro number
REAL(KIND=JPRB) :: RPI
REAL(KIND=JPRB) :: RCLUM
REAL(KIND=JPRB) :: RHPLA
REAL(KIND=JPRB) :: RKBOL
REAL(KIND=JPRB) :: RNAVO

! A1.1 Astronomical constants
! * RDAY         : duration of the solar day
! * RDAYI        : invariant time unit of 86400s
! * RHOUR        : duration of the solar hour
! * REA          : astronomical unit (mean distance Earth-sun)
! * REPSM        : polar axis tilting angle
! * RSIYEA       : duration of the sideral year
! * RSIDAY       : duration of the sideral day
! * ROMEGA       : angular velocity of the Earth rotation
REAL(KIND=JPRB) :: RDAY
REAL(KIND=JPRB) :: RDAYI
REAL(KIND=JPRB) :: RHOUR
REAL(KIND=JPRB) :: REA
REAL(KIND=JPRB) :: REPSM
REAL(KIND=JPRB) :: RSIYEA
REAL(KIND=JPRB) :: RSIDAY
REAL(KIND=JPRB) :: ROMEGA

! A1.2 Geoide
! * RA           : Earth radius
! * RG           : gravity constant
! * R1SA         : 1/RA
REAL(KIND=JPRB) :: RA
REAL(KIND=JPRB) :: RG
REAL(KIND=JPRB) :: R1SA

! A1.3 Radiation
! * RSIGMA       : Stefan-Bolzman constant
! * RI0          : solar constant
REAL(KIND=JPRB) :: RSIGMA
REAL(KIND=JPRB) :: RI0

! A1.4 Thermodynamic gas phase
! * R            : perfect gas constant
! * RMD          : dry air molar mass
! * RMV          : vapour water molar mass
! * RMO3         : ozone molar mass
! * RD           : R_dry (dry air constant)
! * RV           : R_vap (vapour water constant)
! * RCPD         : Cp_dry (dry air calorific capacity at constant pressure)
! * RCPV         : Cp_vap (vapour calorific capacity at constant pressure)
! * RCVD         : Cv_dry (dry air calorific capacity at constant volume)
! * RCVV         : Cv_vap (vapour calorific capacity at constant volume)
! * RKAPPA       : Kappa = R_dry/Cp_dry
! * RETV         : R_vap/R_dry - 1
! * RMCO2        : CO2 (carbon dioxyde) molar mass
! * RMCH4        : CH4 (methane) molar mass
! * RMN2O        : N2O molar mass
! * RMCO         : CO (carbon monoxyde) molar mass
! * RMHCHO       : HCHO molar mass
! * RMNO2        : NO2 (nitrogen dioxyde) molar mass
! * RMSO2        : SO2 (sulfur dioxyde) molar mass
! * RMSO4        : SO4 (sulphate) molar mass
REAL(KIND=JPRB) :: R
REAL(KIND=JPRB) :: RMD
REAL(KIND=JPRB) :: RMV
REAL(KIND=JPRB) :: RMO3
REAL(KIND=JPRB) :: RD
REAL(KIND=JPRB) :: RV
REAL(KIND=JPRB) :: RCPD
REAL(KIND=JPRB) :: RCPV
REAL(KIND=JPRB) :: RCVD
REAL(KIND=JPRB) :: RCVV
REAL(KIND=JPRB) :: RKAPPA
REAL(KIND=JPRB) :: RETV
REAL(KIND=JPRB) :: RMCO2
REAL(KIND=JPRB) :: RMCH4
REAL(KIND=JPRB) :: RMN2O
REAL(KIND=JPRB) :: RMCO
REAL(KIND=JPRB) :: RMHCHO
REAL(KIND=JPRB) :: RMNO2
REAL(KIND=JPRB) :: RMSO2
REAL(KIND=JPRB) :: RMSO4

! A1.5,6 Thermodynamic liquid,solid phases
! * RCW          : Cw (calorific capacity of liquid water)
! * RCS          : Cs (calorific capacity of solid water)
REAL(KIND=JPRB) :: RCW
REAL(KIND=JPRB) :: RCS

! A1.7 Thermodynamic transition of phase
! * RATM         : pre_n = "normal" pressure
! * RTT          : Tt = temperature of water fusion at "pre_n"
! * RLVTT        : RLvTt = vaporisation latent heat at T=Tt
! * RLSTT        : RLsTt = sublimation latent heat at T=Tt
! * RLVZER       : RLv0 = vaporisation latent heat at T=0K
! * RLSZER       : RLs0 = sublimation latent heat at T=0K
! * RLMLT        : RLMlt = melting latent heat at T=Tt
! * RDT          : Tt - Tx(ew-ei)
REAL(KIND=JPRL) :: RATM   ! SC2026: JPRL — values exceed FP16 max (65504)
REAL(KIND=JPRB) :: RTT
REAL(KIND=JPRL) :: RLVTT  ! SC2026: JPRL — values exceed FP16 max (65504)
REAL(KIND=JPRL) :: RLSTT  ! SC2026: JPRL — values exceed FP16 max (65504)
REAL(KIND=JPRB) :: RLVZER
REAL(KIND=JPRB) :: RLSZER
REAL(KIND=JPRL) :: RLMLT  ! SC2026: JPRL — values exceed FP16 max (65504)
REAL(KIND=JPRB) :: RDT

! A1.8 Curve of saturation
! * RESTT        : es(Tt) = saturation vapour tension at T=Tt
! * RGAMW        : Rgamw = (Cw-Cp_vap)/R_vap
! * RBETW        : Rbetw = RLvTt/R_vap + Rgamw*Tt
! * RALPW        : Ralpw = log(es(Tt)) + Rbetw/Tt + Rgamw*log(Tt)
! * RGAMS        : Rgams = (Cs-Cp_vap)/R_vap
! * RBETS        : Rbets = RLsTt/R_vap + Rgams*Tt
! * RALPS        : Ralps = log(es(Tt)) + Rbets/Tt + Rgams*log(Tt)
! * RALPD        : Ralpd = Ralps - Ralpw
! * RBETD        : Rbetd = Rbets - Rbetw
! * RGAMD        : Rgamd = Rgams - Rgamw
REAL(KIND=JPRB) :: RESTT
REAL(KIND=JPRB) :: RGAMW
REAL(KIND=JPRB) :: RBETW
REAL(KIND=JPRB) :: RALPW
REAL(KIND=JPRB) :: RGAMS
REAL(KIND=JPRB) :: RBETS
REAL(KIND=JPRB) :: RALPS
REAL(KIND=JPRB) :: RALPD
REAL(KIND=JPRB) :: RBETD
REAL(KIND=JPRB) :: RGAMD

! NaN value
! CHARACTER(LEN=8), PARAMETER :: CSNAN = &
!   & CHAR(0)//CHAR(0)//CHAR(0)//CHAR(0)//CHAR(0)//CHAR(0)//CHAR(244)//CHAR(127)
REAL(KIND=JPRB) :: RSNAN

!$acc declare copyin(rg, rd, rcpd, retv, rlvtt, rlstt, rlmlt, rtt, rv)
!$omp declare target(rg, rd, rcpd, retv, rlvtt, rlstt, rlmlt, rtt, rv)


!    ------------------------------------------------------------------

TYPE :: TOMCST 
! A1.0 Fundamental constants
! * RPI          : number Pi
! * RCLUM        : light velocity
! * RHPLA        : Planck constant
! * RKBOL        : Bolzmann constant
! * RNAVO        : Avogadro number
REAL(KIND=JPRB) :: RPI
REAL(KIND=JPRB) :: RCLUM
REAL(KIND=JPRB) :: RHPLA
REAL(KIND=JPRB) :: RKBOL
REAL(KIND=JPRB) :: RNAVO

! A1.1 Astronomical constants
! * RDAY         : duration of the solar day
! * RDAYI        : invariant time unit of 86400s
! * RHOUR        : duration of the solar hour 
! * REA          : astronomical unit (mean distance Earth-sun)
! * REPSM        : polar axis tilting angle
! * RSIYEA       : duration of the sideral year
! * RSIDAY       : duration of the sideral day
! * ROMEGA       : angular velocity of the Earth rotation
REAL(KIND=JPRB) :: RDAY
REAL(KIND=JPRB) :: RDAYI
REAL(KIND=JPRB) :: RHOUR
REAL(KIND=JPRB) :: REA
REAL(KIND=JPRB) :: REPSM
REAL(KIND=JPRB) :: RSIYEA
REAL(KIND=JPRB) :: RSIDAY
REAL(KIND=JPRB) :: ROMEGA

! A1.2 Geoide
! * RA           : Earth radius
! * RG           : gravity constant
! * R1SA         : 1/RA
REAL(KIND=JPRB) :: RA
REAL(KIND=JPRB) :: RG
REAL(KIND=JPRB) :: R1SA

! A1.3 Radiation
! * RSIGMA       : Stefan-Bolzman constant
! * RI0          : solar constant
REAL(KIND=JPRB) :: RSIGMA
REAL(KIND=JPRB) :: RI0

! A1.4 Thermodynamic gas phase
! * R            : perfect gas constant
! * RMD          : dry air molar mass
! * RMV          : vapour water molar mass
! * RMO3         : ozone molar mass
! * RD           : R_dry (dry air constant)
! * RV           : R_vap (vapour water constant)
! * RCPD         : Cp_dry (dry air calorific capacity at constant pressure)
! * RCPV         : Cp_vap (vapour calorific capacity at constant pressure)
! * RCVD         : Cv_dry (dry air calorific capacity at constant volume)
! * RCVV         : Cv_vap (vapour calorific capacity at constant volume)
! * RKAPPA       : Kappa = R_dry/Cp_dry
! * RETV         : R_vap/R_dry - 1
! * RMCO2        : CO2 (carbon dioxyde) molar mass
! * RMCH4        : CH4 (methane) molar mass
! * RMN2O        : N2O molar mass
! * RMCO         : CO (carbon monoxyde) molar mass
! * RMHCHO       : HCHO molar mass
! * RMNO2        : NO2 (nitrogen dioxyde) molar mass
! * RMSO2        : SO2 (sulfur dioxyde) molar mass
! * RMSO4        : SO4 (sulphate) molar mass
REAL(KIND=JPRB) :: R
REAL(KIND=JPRB) :: RMD
REAL(KIND=JPRB) :: RMV
REAL(KIND=JPRB) :: RMO3
REAL(KIND=JPRB) :: RD
REAL(KIND=JPRB) :: RV
REAL(KIND=JPRB) :: RCPD
REAL(KIND=JPRB) :: RCPV
REAL(KIND=JPRB) :: RCVD
REAL(KIND=JPRB) :: RCVV
REAL(KIND=JPRB) :: RKAPPA
REAL(KIND=JPRB) :: RETV
REAL(KIND=JPRB) :: RMCO2
REAL(KIND=JPRB) :: RMCH4
REAL(KIND=JPRB) :: RMN2O
REAL(KIND=JPRB) :: RMCO
REAL(KIND=JPRB) :: RMHCHO
REAL(KIND=JPRB) :: RMNO2
REAL(KIND=JPRB) :: RMSO2
REAL(KIND=JPRB) :: RMSO4

! A1.5,6 Thermodynamic liquid,solid phases
! * RCW          : Cw (calorific capacity of liquid water)
! * RCS          : Cs (calorific capacity of solid water)
REAL(KIND=JPRB) :: RCW
REAL(KIND=JPRB) :: RCS

! A1.7 Thermodynamic transition of phase
! * RATM         : pre_n = "normal" pressure
! * RTT          : Tt = temperature of water fusion at "pre_n"
! * RLVTT        : RLvTt = vaporisation latent heat at T=Tt
! * RLSTT        : RLsTt = sublimation latent heat at T=Tt
! * RLVZER       : RLv0 = vaporisation latent heat at T=0K
! * RLSZER       : RLs0 = sublimation latent heat at T=0K
! * RLMLT        : RLMlt = melting latent heat at T=Tt
! * RDT          : Tt - Tx(ew-ei)
REAL(KIND=JPRL) :: RATM   ! SC2026: JPRL — values exceed FP16 max (65504)
REAL(KIND=JPRB) :: RTT
REAL(KIND=JPRL) :: RLVTT  ! SC2026: JPRL — values exceed FP16 max (65504)
REAL(KIND=JPRL) :: RLSTT  ! SC2026: JPRL — values exceed FP16 max (65504)
REAL(KIND=JPRB) :: RLVZER
REAL(KIND=JPRB) :: RLSZER
REAL(KIND=JPRL) :: RLMLT  ! SC2026: JPRL — values exceed FP16 max (65504)
REAL(KIND=JPRB) :: RDT

! A1.8 Curve of saturation
! * RESTT        : es(Tt) = saturation vapour tension at T=Tt
! * RGAMW        : Rgamw = (Cw-Cp_vap)/R_vap
! * RBETW        : Rbetw = RLvTt/R_vap + Rgamw*Tt
! * RALPW        : Ralpw = log(es(Tt)) + Rbetw/Tt + Rgamw*log(Tt)
! * RGAMS        : Rgams = (Cs-Cp_vap)/R_vap
! * RBETS        : Rbets = RLsTt/R_vap + Rgams*Tt
! * RALPS        : Ralps = log(es(Tt)) + Rbets/Tt + Rgams*log(Tt)
! * RALPD        : Ralpd = Ralps - Ralpw
! * RBETD        : Rbetd = Rbets - Rbetw
! * RGAMD        : Rgamd = Rgams - Rgamw
REAL(KIND=JPRB) :: RESTT
REAL(KIND=JPRB) :: RGAMW
REAL(KIND=JPRB) :: RBETW
REAL(KIND=JPRB) :: RALPW
REAL(KIND=JPRB) :: RGAMS
REAL(KIND=JPRB) :: RBETS
REAL(KIND=JPRB) :: RALPS
REAL(KIND=JPRB) :: RALPD
REAL(KIND=JPRB) :: RBETD
REAL(KIND=JPRB) :: RGAMD

! NaN value
! CHARACTER(LEN=8), PARAMETER :: CSNAN = &
!   & CHAR(0)//CHAR(0)//CHAR(0)//CHAR(0)//CHAR(0)//CHAR(0)//CHAR(244)//CHAR(127)
REAL(KIND=JPRB) :: RSNAN

END TYPE TOMCST

TYPE(TOMCST), ALLOCATABLE :: YRCST
END MODULE YOMCST

! (C) Copyright 1988- ECMWF.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
!
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation
! nor does it submit to any jurisdiction.

MODULE YOETHF

USE PARKIND1,    ONLY : JPIM, JPRB, JPRL


IMPLICIT NONE

SAVE

!     ------------------------------------------------------------------
!*     *YOETHF* DERIVED CONSTANTS SPECIFIC TO ECMWF THERMODYNAMICS
!     ------------------------------------------------------------------

REAL(KIND=JPRB) :: R2ES
REAL(KIND=JPRB) :: R3LES
REAL(KIND=JPRB) :: R3IES
REAL(KIND=JPRB) :: R4LES
REAL(KIND=JPRB) :: R4IES
REAL(KIND=JPRB) :: R5LES
REAL(KIND=JPRB) :: R5IES
REAL(KIND=JPRB) :: RVTMP2
REAL(KIND=JPRB) :: RHOH2O
REAL(KIND=JPRL) :: R5ALVCP  ! SC2026: JPRL — values exceed FP16 max (65504)
REAL(KIND=JPRL) :: R5ALSCP  ! SC2026: JPRL — values exceed FP16 max (65504)
REAL(KIND=JPRB) :: RALVDCP
REAL(KIND=JPRB) :: RALSDCP
REAL(KIND=JPRB) :: RALFDCP
REAL(KIND=JPRB) :: RTWAT
REAL(KIND=JPRB) :: RTBER
REAL(KIND=JPRB) :: RTBERCU
REAL(KIND=JPRB) :: RTICE
REAL(KIND=JPRB) :: RTICECU
REAL(KIND=JPRB) :: RTWAT_RTICE_R
REAL(KIND=JPRB) :: RTWAT_RTICECU_R
REAL(KIND=JPRB) :: RKOOP1
REAL(KIND=JPRB) :: RKOOP2

TYPE :: TOETHF
REAL(KIND=JPRB) :: R2ES
REAL(KIND=JPRB) :: R3LES
REAL(KIND=JPRB) :: R3IES
REAL(KIND=JPRB) :: R4LES
REAL(KIND=JPRB) :: R4IES
REAL(KIND=JPRB) :: R5LES
REAL(KIND=JPRB) :: R5IES
REAL(KIND=JPRB) :: RVTMP2
REAL(KIND=JPRB) :: RHOH2O
REAL(KIND=JPRL) :: R5ALVCP  ! SC2026: JPRL — values exceed FP16 max (65504)
REAL(KIND=JPRL) :: R5ALSCP  ! SC2026: JPRL — values exceed FP16 max (65504)
REAL(KIND=JPRB) :: RALVDCP
REAL(KIND=JPRB) :: RALSDCP
REAL(KIND=JPRB) :: RALFDCP
REAL(KIND=JPRB) :: RTWAT
REAL(KIND=JPRB) :: RTBER
REAL(KIND=JPRB) :: RTBERCU
REAL(KIND=JPRB) :: RTICE
REAL(KIND=JPRB) :: RTICECU
REAL(KIND=JPRB) :: RTWAT_RTICE_R
REAL(KIND=JPRB) :: RTWAT_RTICECU_R
REAL(KIND=JPRB) :: RKOOP1
REAL(KIND=JPRB) :: RKOOP2
END TYPE TOETHF

TYPE(TOETHF), ALLOCATABLE :: YRTHF

!     J.-J. MORCRETTE                   91/07/14  ADAPTED TO I.F.S.

!      NAME     TYPE      PURPOSE
!      ----     ----      -------

!     *R__ES*   REAL      *CONSTANTS USED FOR COMPUTATION OF SATURATION
!                         MIXING RATIO OVER LIQUID WATER(*R_LES*) OR
!                         ICE(*R_IES*).
!     *RVTMP2*  REAL      *RVTMP2=RCPV/RCPD-1.
!     *RHOH2O*  REAL      *DENSITY OF LIQUID WATER.   (RATM/100.)
!     *R5ALVCP* REAL      *R5LES*RLVTT/RCPD
!     *R5ALSCP* REAL      *R5IES*RLSTT/RCPD
!     *RALVDCP* REAL      *RLVTT/RCPD
!     *RALSDCP* REAL      *RLSTT/RCPD
!     *RALFDCP* REAL      *RLMLT/RCPD
!     *RTWAT*   REAL      *RTWAT=RTT
!     *RTBER*   REAL      *RTBER=RTT-0.05
!     *RTBERCU  REAL      *RTBERCU=RTT-5.0
!     *RTICE*   REAL      *RTICE=RTT-0.1
!     *RTICECU* REAL      *RTICECU=RTT-23.0
!     *RKOOP?   REAL      *CONSTANTS TO DESCRIBE KOOP FORM FOR NUCLEATION
!     *RTWAT_RTICE_R*   REAL      *RTWAT_RTICE_R=1./(RTWAT-RTICE)
!     *RTWAT_RTICECU_R* REAL      *RTWAT_RTICECU_R=1./(RTWAT-RTICECU)

!$acc declare copyin(r2es, r3les, r3ies, r4les, r4ies, r5les, r5ies, &
!$acc   r5alvcp, r5alscp, ralvdcp, ralsdcp, ralfdcp, rtwat, rtice, rticecu, &
!$acc   rtwat_rtice_r, rtwat_rticecu_r, rkoop1, rkoop2)

!$omp declare target(r2es, r3les, r3ies, r4les, r4ies, r5les, r5ies)
!$omp declare target(  r5alvcp, r5alscp, ralvdcp, ralsdcp, ralfdcp, rtwat, rtice, rticecu)
!$omp declare target(  rtwat_rtice_r, rtwat_rticecu_r, rkoop1, rkoop2)

!       ----------------------------------------------------------------
END MODULE YOETHF

! (C) Copyright 1988- ECMWF.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
!
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation
! nor does it submit to any jurisdiction.

MODULE YOECLDP

USE PARKIND1,    ONLY : JPIM, JPRB, JPRL


IMPLICIT NONE

SAVE

!     -----------------------------------------------------------------
!     ** YOECLDP - CONTROL PARAMETERS FOR PROGNOSTIC CLOUD SCHEME
!     -----------------------------------------------------------------

!     * E.C.M.W.F. PHYSICS PACKAGE *

!     C. JAKOB     E.C.M.W.F.    94/02/07
!     A. Tompkins  E.C.M.W.F.  2004/12/03 total water variance setup for
!                                                  moist advection-diffusion PBL
!     A. Tompkins  E.C.M.W.F.  2004/09/02 Aerosol in microphysics switches
!     JJMorcrette  ECMWF       20100813   Aerosol index for aerosol-cloud interactions
!     R. Forbes    ECMWF       20110301   Added ice deposition parameters
!     R. Forbes    ECMWF       20150115   Added additional ice, snow and rain parameters

!      NAME     TYPE      PURPOSE
!      ----     ----      -------

!     *RAMID*   REAL      BASE VALUE FOR CALCULATION OF RELATIVE 
!                         HUMIDITY THRESHOLD FOR ONSET OF STRATIFORM
!                         CONDENSATION (TIEDTKE, 1993, EQUATION 24)
!     *RCLDIFF* REAL      DIFFUSION-COEFFICIENT FOR EVAPORATION BY
!                         TURBULENT MIXING (IBID., EQU. 30)
!     *RCLDIFF_CONVI*REAL ENHANCEMENT FACTOR OF  RCLDIFF FOR CONVECTION
!     *RCLCRIT* REAL      BASE VALUE OF CRITICAL CLOUD WATER CONTENT 
!                         FOR CONVERSION TO RAIN (SUNDQUIST, 1988)
!     *RCLCRIT_SEA* REAL  BASE VALUE OF CRITICAL CLOUD WATER CONTENT FOR SEA
!     *RCLCRIT_LAND* REAL BASE VALUE OF CRITICAL CLOUD WATER CONTENT FOR LAND
!     *RKCONV*  REAL      BASE VALUE FOR CONVERSION COEFFICIENT (IBID.)
!     *RPRC1*   REAL      COALESCENCE CONSTANT (IBID.)
!     *RPRC2*   REAL      BERGERON-FINDEISEN CONSTANT (IBID.)
!     *RCLDMAX* REAL      MAXIMUM CLOUD WATER CONTENT
!     *RPECONS* REAL      EVAPORATION CONSTANT AFTER KESSLER 
!                         (TIEDTKE, 1993, EQU.35)
!     *RPRECRHMAX* REAL   MAX THRESHOLD RH FOR EVAPORATION FOR ZERO COVER
!     *RTAUMEL* REAL      RELAXATION TIME FOR MELTING OF SNOW
!     *RAMIN*   REAL      LIMIT FOR A
!     *RLMIN*   REAL      LIMIT FOR L
!     *RKOOPTAU*  REAL    TIMESCALE FOR ICE SUPERSATURATION REMOVAL
!     *RVICE*     REAL    FIXED ICE FALLSPEED
!     *RVRAIN*    REAL    FIXED RAIN FALLSPEED
!     *RVSNOW*    REAL    FIXED SNOW FALLSPEED
!     *RTHOMO*    REAL    TEMPERATURE THRESHOLD FOR SPONTANEOUS FREEZING OF LIQUID DROPLETS
!     *RCOVPMIN*  REAL    MINIMUM PRECIPITATION COVERAGE REQUIRED FOR THE NEW PROGNOSTIC PRECIP
!     *RCLDTOPP*  REAL    TOP PRESSURE FOR CLOUD CALCULATION
!     *NCLDTOP*   INTEGER TOP LEVEL FOR CLOUD CALCULATION
!     *NSSOPT*  INTEGER   PARAMETRIZATION CHOICE FOR SUPERSATURATION
!     *NCLDDIAG*INTEGER   CONTROLS CLOUDSC DIAGNOSTICS IN PEXTRA
!     *NCLV*     INTEGER   NUMBER OF PROGNOSTIC EQUATIONS IN CLOUDSC 
!                         (INCLUDES WATER VAPOUR AS DUMMY VARIABLE) 
!      NAERCLD         INT  INDEX TO CONTROL SWITCHES FOR 
!                           AEROSOL-MICROPHYSICS INTERACTION, LAER*
!      NAECLxx         INT  INDEX OF GEMS AEROSOLS USED IN AEROSOL-CLOUD INTERACTIONS
!      RCCN            REAL DEFAULT CCN (CM-3)
!      RNICE           REAL DEFAULT ICE NUMBER CONCENTRATION (CM-3)
!      LAERLIQAUTOLSP  LOG  AEROSOLS AFFECT RAIN AUTOCONVERSION IN LSP
!      LAERLIQAUTOCP   LOG  AEROSOLS AFFECT RAIN AUTOCONVERSION IN CP
!      LAERLIQCOLL     LOG  AEROSOLS AFFECT RAIN COLLECTION 
!      LAERICESED      LOG  AEROSOLS AFFECT ICE SEDIMENTATION
!      LAERICEAUTO     LOG  AEROSOLS AFFECT ICE AUTOCONVERSION
!      RCCNOM          REAL CONSTANT IN MENON PARAM FOR ORGANIC MATTER -> CCN
!      RCCNSS          REAL CONSTANT IN MENON PARAM SEA SALT -> CCN
!      RCCNSU          REAL CONSTANT IN MENON PARAM FOR SULPHATE -> CCN
!      RCLDTOPCF       REAL Cloud fraction threshold that defines cloud top 
!      RDEPLIQREFRATE  REAL Fraction of deposition rate in cloud top layer
!      RDEPLIQREFDEPTH REAL Depth of supercooled liquid water layer (m)
!      RVRFACTOR       REAL KESSLER FACTOR=5.09E-3 FOR EVAPORATION OF CLEAR-SKY RAIN  (KESSLER,1969)

INTEGER(KIND=JPIM),PARAMETER :: NCLV=5      ! number of microphysics variables
INTEGER(KIND=JPIM),PARAMETER :: NCLDQL=1    ! liquid cloud water
INTEGER(KIND=JPIM),PARAMETER :: NCLDQI=2    ! ice cloud water
INTEGER(KIND=JPIM),PARAMETER :: NCLDQR=3    ! rain water
INTEGER(KIND=JPIM),PARAMETER :: NCLDQS=4    ! snow
INTEGER(KIND=JPIM),PARAMETER :: NCLDQV=5    ! vapour


TYPE :: TECLDP
REAL(KIND=JPRL) :: RAMID
REAL(KIND=JPRL) :: RCLDIFF
REAL(KIND=JPRL) :: RCLDIFF_CONVI
REAL(KIND=JPRL) :: RCLCRIT
REAL(KIND=JPRL) :: RCLCRIT_SEA
REAL(KIND=JPRL) :: RCLCRIT_LAND
REAL(KIND=JPRL) :: RKCONV
REAL(KIND=JPRL) :: RPRC1
REAL(KIND=JPRL) :: RPRC2
REAL(KIND=JPRL) :: RCLDMAX
REAL(KIND=JPRL) :: RPECONS
REAL(KIND=JPRL) :: RVRFACTOR
REAL(KIND=JPRL) :: RPRECRHMAX
REAL(KIND=JPRL) :: RTAUMEL
REAL(KIND=JPRL) :: RAMIN
REAL(KIND=JPRL) :: RLMIN
REAL(KIND=JPRL) :: RKOOPTAU
REAL(KIND=JPRL) :: RCLDTOPP
REAL(KIND=JPRL) :: RLCRITSNOW
REAL(KIND=JPRL) :: RSNOWLIN1
REAL(KIND=JPRL) :: RSNOWLIN2
REAL(KIND=JPRL) :: RICEHI1
REAL(KIND=JPRL) :: RICEHI2
REAL(KIND=JPRL) :: RICEINIT
REAL(KIND=JPRL) :: RVICE
REAL(KIND=JPRL) :: RVRAIN
REAL(KIND=JPRL) :: RVSNOW
REAL(KIND=JPRL) :: RTHOMO
REAL(KIND=JPRL) :: RCOVPMIN
REAL(KIND=JPRL) :: RCCN
REAL(KIND=JPRL) :: RNICE
REAL(KIND=JPRL) :: RCCNOM
REAL(KIND=JPRL) :: RCCNSS
REAL(KIND=JPRL) :: RCCNSU
REAL(KIND=JPRL) :: RCLDTOPCF
REAL(KIND=JPRL) :: RDEPLIQREFRATE
REAL(KIND=JPRL) :: RDEPLIQREFDEPTH
!--------------------------------------------------------
! Autoconversion/accretion (Khairoutdinov and Kogan 2000)
!--------------------------------------------------------
REAL(KIND=JPRL) :: RCL_KKAac
REAL(KIND=JPRL) :: RCL_KKBac
REAL(KIND=JPRL) :: RCL_KKAau
REAL(KIND=JPRL) :: RCL_KKBauq
REAL(KIND=JPRL) :: RCL_KKBaun
REAL(KIND=JPRL) :: RCL_KK_cloud_num_sea
REAL(KIND=JPRL) :: RCL_KK_cloud_num_land
!--------------------------------------------------------
! Ice
!--------------------------------------------------------
REAL(KIND=JPRL) :: RCL_AI
REAL(KIND=JPRL) :: RCL_BI
REAL(KIND=JPRL) :: RCL_CI
REAL(KIND=JPRL) :: RCL_DI
REAL(KIND=JPRL) :: RCL_X1I  ! SC2026: JPRL — value(s) exceed FP16 max
REAL(KIND=JPRL) :: RCL_X2I
REAL(KIND=JPRL) :: RCL_X3I
REAL(KIND=JPRL) :: RCL_X4I
REAL(KIND=JPRL) :: RCL_CONST1I
REAL(KIND=JPRL) :: RCL_CONST2I  ! SC2026: JPRL — value(s) exceed FP16 max
REAL(KIND=JPRL) :: RCL_CONST3I
REAL(KIND=JPRL) :: RCL_CONST4I
REAL(KIND=JPRL) :: RCL_CONST5I
REAL(KIND=JPRL) :: RCL_CONST6I
REAL(KIND=JPRL) :: RCL_APB1  ! SC2026: JPRL — value(s) exceed FP16 max
REAL(KIND=JPRL) :: RCL_APB2  ! SC2026: JPRL — value(s) exceed FP16 max
REAL(KIND=JPRL) :: RCL_APB3
!--------------------------------------------------------
! Snow
!--------------------------------------------------------
REAL(KIND=JPRL) :: RCL_AS
REAL(KIND=JPRL) :: RCL_BS
REAL(KIND=JPRL) :: RCL_CS
REAL(KIND=JPRL) :: RCL_DS
REAL(KIND=JPRL) :: RCL_X1S  ! SC2026: JPRL — value(s) exceed FP16 max
REAL(KIND=JPRL) :: RCL_X2S
REAL(KIND=JPRL) :: RCL_X3S
REAL(KIND=JPRL) :: RCL_X4S
REAL(KIND=JPRL) :: RCL_CONST1S
REAL(KIND=JPRL) :: RCL_CONST2S  ! SC2026: JPRL — value(s) exceed FP16 max
REAL(KIND=JPRL) :: RCL_CONST3S
REAL(KIND=JPRL) :: RCL_CONST4S
REAL(KIND=JPRL) :: RCL_CONST5S
REAL(KIND=JPRL) :: RCL_CONST6S
REAL(KIND=JPRL) :: RCL_CONST7S  ! SC2026: JPRL — value(s) exceed FP16 max
REAL(KIND=JPRL) :: RCL_CONST8S
!--------------------------------------------------------
! Rain
!--------------------------------------------------------
REAL(KIND=JPRL) :: RDENSWAT
REAL(KIND=JPRL) :: RDENSREF
REAL(KIND=JPRL) :: RCL_AR
REAL(KIND=JPRL) :: RCL_BR
REAL(KIND=JPRL) :: RCL_CR
REAL(KIND=JPRL) :: RCL_DR
REAL(KIND=JPRL) :: RCL_X1R
REAL(KIND=JPRL) :: RCL_X2R
REAL(KIND=JPRL) :: RCL_X4R
REAL(KIND=JPRL) :: RCL_KA273
REAL(KIND=JPRL) :: RCL_CDENOM1  ! SC2026: JPRL — value(s) exceed FP16 max
REAL(KIND=JPRL) :: RCL_CDENOM2  ! SC2026: JPRL — value(s) exceed FP16 max
REAL(KIND=JPRL) :: RCL_CDENOM3
REAL(KIND=JPRL) :: RCL_SCHMIDT
REAL(KIND=JPRL) :: RCL_DYNVISC
REAL(KIND=JPRL) :: RCL_CONST1R
REAL(KIND=JPRL) :: RCL_CONST2R  ! SC2026: JPRL — value(s) exceed FP16 max
REAL(KIND=JPRL) :: RCL_CONST3R
REAL(KIND=JPRL) :: RCL_CONST4R
REAL(KIND=JPRL) :: RCL_FAC1
REAL(KIND=JPRL) :: RCL_FAC2
! Rain freezing
REAL(KIND=JPRL) :: RCL_CONST5R  ! SC2026: JPRL — value(s) exceed FP16 max
REAL(KIND=JPRL) :: RCL_CONST6R
REAL(KIND=JPRL) :: RCL_FZRAB
REAL(KIND=JPRL) :: RCL_FZRBB

LOGICAL :: LCLDEXTRA, LCLDBUDGET

INTEGER(KIND=JPIM) :: NSSOPT
INTEGER(KIND=JPIM) :: NCLDTOP
INTEGER(KIND=JPIM) :: NAECLBC, NAECLDU, NAECLOM, NAECLSS, NAECLSU
INTEGER(KIND=JPIM) :: NCLDDIAG

! aerosols
INTEGER(KIND=JPIM) :: NAERCLD
LOGICAL :: LAERLIQAUTOLSP
LOGICAL :: LAERLIQAUTOCP
LOGICAL :: LAERLIQAUTOCPB
LOGICAL :: LAERLIQCOLL
LOGICAL :: LAERICESED
LOGICAL :: LAERICEAUTO

! variance arrays
REAL(KIND=JPRL) :: NSHAPEP
REAL(KIND=JPRL) :: NSHAPEQ
INTEGER(KIND=JPIM) :: NBETA
REAL(KIND=JPRL) :: RBETA(0:100)
REAL(KIND=JPRL) :: RBETAP1(0:100)


END TYPE TECLDP

TYPE(TECLDP), ALLOCATABLE :: YRECLDP
END MODULE YOECLDP

MODULE YOMPHYDER
  USE PARKIND1, ONLY : JPIM, JPRB
  IMPLICIT NONE
  TYPE :: STATE_TYPE
    REAL(KIND=JPRB), POINTER :: U(:,:) => NULL()
  END TYPE STATE_TYPE
END MODULE YOMPHYDER