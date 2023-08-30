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
#ifdef SINGLE
INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(6,37)
#else
INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13,300)
#endif

! Double real for C code and special places requiring 
!    higher precision. 
INTEGER, PARAMETER :: JPRD = SELECTED_REAL_KIND(13,300)


! Logical Kinds for RTTOV....

INTEGER, PARAMETER :: JPLM = JPIM   !Standard logical type

END MODULE PARKIND1


MODULE YOMCST

USE PARKIND1,    ONLY : JPRB

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
REAL(KIND=JPRB) :: RATM
REAL(KIND=JPRB) :: RTT
REAL(KIND=JPRB) :: RLVTT
REAL(KIND=JPRB) :: RLSTT
REAL(KIND=JPRB) :: RLVZER
REAL(KIND=JPRB) :: RLSZER
REAL(KIND=JPRB) :: RLMLT
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


END MODULE YOMCST


MODULE YOETHF

USE PARKIND1,    ONLY : JPIM, JPRB

IMPLICIT NONE

SAVE

REAL(KIND=JPRB) :: R2ES
REAL(KIND=JPRB) :: R3LES
REAL(KIND=JPRB) :: R3IES
REAL(KIND=JPRB) :: R4LES
REAL(KIND=JPRB) :: R4IES
REAL(KIND=JPRB) :: R5LES
REAL(KIND=JPRB) :: R5IES
REAL(KIND=JPRB) :: RVTMP2
REAL(KIND=JPRB) :: RHOH2O
REAL(KIND=JPRB) :: R5ALVCP
REAL(KIND=JPRB) :: R5ALSCP
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


END MODULE YOETHF


MODULE YOECLDP

USE PARKIND1,    ONLY : JPIM, JPRB
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
INTEGER(KIND=JPIM),PARAMETER :: NCLDQV=25    ! vapour

REAL(KIND=JPRB) :: RAMID
REAL(KIND=JPRB) :: RCLDIFF
REAL(KIND=JPRB) :: RCLDIFF_CONVI
REAL(KIND=JPRB) :: RCLCRIT
REAL(KIND=JPRB) :: RCLCRIT_SEA
REAL(KIND=JPRB) :: RCLCRIT_LAND
REAL(KIND=JPRB) :: RKCONV
REAL(KIND=JPRB) :: RPRC1
REAL(KIND=JPRB) :: RPRC2
REAL(KIND=JPRB) :: RCLDMAX
REAL(KIND=JPRB) :: RPECONS
REAL(KIND=JPRB) :: RVRFACTOR
REAL(KIND=JPRB) :: RPRECRHMAX
REAL(KIND=JPRB) :: RTAUMEL
REAL(KIND=JPRB) :: RAMIN
REAL(KIND=JPRB) :: RLMIN
REAL(KIND=JPRB) :: RKOOPTAU
REAL(KIND=JPRB) :: RCLDTOPP
REAL(KIND=JPRB) :: RLCRITSNOW
REAL(KIND=JPRB) :: RSNOWLIN1
REAL(KIND=JPRB) :: RSNOWLIN2
REAL(KIND=JPRB) :: RICEHI1
REAL(KIND=JPRB) :: RICEHI2
REAL(KIND=JPRB) :: RICEINIT
REAL(KIND=JPRB) :: RVICE
REAL(KIND=JPRB) :: RVRAIN
REAL(KIND=JPRB) :: RVSNOW
REAL(KIND=JPRB) :: RTHOMO
REAL(KIND=JPRB) :: RCOVPMIN
REAL(KIND=JPRB) :: RCCN
REAL(KIND=JPRB) :: RNICE
REAL(KIND=JPRB) :: RCCNOM
REAL(KIND=JPRB) :: RCCNSS
REAL(KIND=JPRB) :: RCCNSU
REAL(KIND=JPRB) :: RCLDTOPCF
REAL(KIND=JPRB) :: RDEPLIQREFRATE
REAL(KIND=JPRB) :: RDEPLIQREFDEPTH
!--------------------------------------------------------
! Autoconversion/accretion (Khairoutdinov and Kogan 2000)
!--------------------------------------------------------
REAL(KIND=JPRB) :: RCL_KKAac
REAL(KIND=JPRB) :: RCL_KKBac
REAL(KIND=JPRB) :: RCL_KKAau
REAL(KIND=JPRB) :: RCL_KKBauq
REAL(KIND=JPRB) :: RCL_KKBaun
REAL(KIND=JPRB) :: RCL_KK_CLOUD_NUM_LAND
REAL(KIND=JPRB) :: RCL_KK_CLOUD_NUM_SEA
!--------------------------------------------------------
! Ice
!--------------------------------------------------------
REAL(KIND=JPRB) :: RCL_AI
REAL(KIND=JPRB) :: RCL_BI
REAL(KIND=JPRB) :: RCL_CI
REAL(KIND=JPRB) :: RCL_DI
REAL(KIND=JPRB) :: RCL_X1I
REAL(KIND=JPRB) :: RCL_X2I
REAL(KIND=JPRB) :: RCL_X3I
REAL(KIND=JPRB) :: RCL_X4I
REAL(KIND=JPRB) :: RCL_CONST1I
REAL(KIND=JPRB) :: RCL_CONST2I
REAL(KIND=JPRB) :: RCL_CONST3I
REAL(KIND=JPRB) :: RCL_CONST4I
REAL(KIND=JPRB) :: RCL_CONST5I
REAL(KIND=JPRB) :: RCL_CONST6I
REAL(KIND=JPRB) :: RCL_APB1
REAL(KIND=JPRB) :: RCL_APB2
REAL(KIND=JPRB) :: RCL_APB3
!--------------------------------------------------------
! Snow
!--------------------------------------------------------
REAL(KIND=JPRB) :: RCL_AS
REAL(KIND=JPRB) :: RCL_BS
REAL(KIND=JPRB) :: RCL_CS
REAL(KIND=JPRB) :: RCL_DS
REAL(KIND=JPRB) :: RCL_X1S
REAL(KIND=JPRB) :: RCL_X2S
REAL(KIND=JPRB) :: RCL_X3S
REAL(KIND=JPRB) :: RCL_X4S
REAL(KIND=JPRB) :: RCL_CONST1S
REAL(KIND=JPRB) :: RCL_CONST2S
REAL(KIND=JPRB) :: RCL_CONST3S
REAL(KIND=JPRB) :: RCL_CONST4S
REAL(KIND=JPRB) :: RCL_CONST5S
REAL(KIND=JPRB) :: RCL_CONST6S
REAL(KIND=JPRB) :: RCL_CONST7S
REAL(KIND=JPRB) :: RCL_CONST8S
!--------------------------------------------------------
! Rain
!--------------------------------------------------------
REAL(KIND=JPRB) :: RDENSWAT
REAL(KIND=JPRB) :: RDENSREF
REAL(KIND=JPRB) :: RCL_AR
REAL(KIND=JPRB) :: RCL_BR
REAL(KIND=JPRB) :: RCL_CR
REAL(KIND=JPRB) :: RCL_DR
REAL(KIND=JPRB) :: RCL_X1R
REAL(KIND=JPRB) :: RCL_X2R
REAL(KIND=JPRB) :: RCL_X4R
REAL(KIND=JPRB) :: RCL_KA273
REAL(KIND=JPRB) :: RCL_CDENOM1
REAL(KIND=JPRB) :: RCL_CDENOM2
REAL(KIND=JPRB) :: RCL_CDENOM3
REAL(KIND=JPRB) :: RCL_SCHMIDT
REAL(KIND=JPRB) :: RCL_DYNVISC
REAL(KIND=JPRB) :: RCL_CONST1R
REAL(KIND=JPRB) :: RCL_CONST2R
REAL(KIND=JPRB) :: RCL_CONST3R
REAL(KIND=JPRB) :: RCL_CONST4R
REAL(KIND=JPRB) :: RCL_FAC1
REAL(KIND=JPRB) :: RCL_FAC2
! Rain freezing
REAL(KIND=JPRB) :: RCL_CONST5R
REAL(KIND=JPRB) :: RCL_CONST6R
REAL(KIND=JPRB) :: RCL_FZRAB
REAL(KIND=JPRB) :: RCL_FZRBB

LOGICAL :: LCLDEXTRA, LCLDBUDGET

INTEGER(KIND=JPIM) :: NSSOPT
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
REAL(KIND=JPRB) :: NSHAPEP
REAL(KIND=JPRB) :: NSHAPEQ
INTEGER(KIND=JPIM) :: NBETA
REAL(KIND=JPRB) :: RBETA !(0:100)
REAL(KIND=JPRB) :: RBETAP1 !(0:100)


END MODULE YOECLDP


PROGRAM CLOUDPROGRAM
  USE PARKIND1 , ONLY : JPIM, JPRB
  IMPLICIT NONE
  SAVE

  INTEGER(KIND=JPIM),PARAMETER    :: KLON =25            ! Number of grid points
  INTEGER(KIND=JPIM),PARAMETER    :: KLEV  =25           ! Number of levels
  INTEGER(KIND=JPIM),PARAMETER    :: NCLV  =25           ! Number of levels
  INTEGER(KIND=JPIM),PARAMETER    :: NBLOCKS = 512
  INTEGER(KIND=JPIM),PARAMETER    :: NCLDTOP = 25

  REAL(KIND=JPRB)       :: PLCRIT_AER(KLON,KLEV,NBLOCKS) 
  REAL(KIND=JPRB)       :: PICRIT_AER(KLON,KLEV,NBLOCKS) 
  REAL(KIND=JPRB)       :: PRE_ICE(KLON,KLEV,NBLOCKS) 
  REAL(KIND=JPRB)       :: PCCN(KLON,KLEV,NBLOCKS)     ! liquid cloud condensation nuclei
  REAL(KIND=JPRB)       :: PNICE(KLON,KLEV,NBLOCKS)    ! ice number concentration (cf. CCN)
  
  REAL(KIND=JPRB)       :: PTSPHY            ! Physics timestep
  REAL(KIND=JPRB)       :: PT(KLON,KLEV,NBLOCKS)    ! T at start of callpar
  REAL(KIND=JPRB)       :: PQ(KLON,KLEV,NBLOCKS)    ! Q at start of callpar
  
  REAL(KIND=JPRB) :: tendency_cml_T(KLON,KLEV,NBLOCKS)    ! GMV fields
  REAL(KIND=JPRB):: tendency_cml_q(KLON,KLEV,NBLOCKS) ,tendency_cml_a(KLON,KLEV,NBLOCKS)   ! GFL fields
  REAL(KIND=JPRB):: tendency_cml_cld(KLON,KLEV,NCLV,NBLOCKS)    ! composed cloud array
  
  REAL(KIND=JPRB):: tendency_tmp_T(KLON,KLEV,NBLOCKS)   ! GMV fields
  REAL(KIND=JPRB):: tendency_tmp_q(KLON,KLEV,NBLOCKS),tendency_tmp_a(KLON,KLEV,NBLOCKS)  ! GFL fields
  REAL(KIND=JPRB)  :: tendency_tmp_cld(KLON,KLEV,NCLV,NBLOCKS)   ! composed cloud array
  
  REAL(KIND=JPRB):: tendency_loc_T(KLON,KLEV,NBLOCKS)   ! GMV fields
  REAL(KIND=JPRB):: tendency_loc_q(KLON,KLEV,NBLOCKS),tendency_loc_a(KLON,KLEV,NBLOCKS)  ! GFL fields
  REAL(KIND=JPRB) :: tendency_loc_cld(KLON,KLEV,NCLV,NBLOCKS)   ! composed cloud array
  
  !TYPE (STATE_TYPE) , INTENT (IN)  :: tendency_cml   ! cumulative tendency used for final output
  !TYPE (STATE_TYPE) , INTENT (IN)  :: tendency_tmp   ! cumulative tendency used as input
  !TYPE (STATE_TYPE) , INTENT (OUT) :: tendency_loc   ! local tendency from cloud scheme
  REAL(KIND=JPRB)       :: PVFA(KLON,KLEV,NBLOCKS)  ! CC from VDF scheme
  REAL(KIND=JPRB)       :: PVFL(KLON,KLEV,NBLOCKS)  ! Liq from VDF scheme
  REAL(KIND=JPRB)       :: PVFI(KLON,KLEV,NBLOCKS)  ! Ice from VDF scheme
  REAL(KIND=JPRB)       :: PDYNA(KLON,KLEV,NBLOCKS) ! CC from Dynamics
  REAL(KIND=JPRB)       :: PDYNL(KLON,KLEV,NBLOCKS) ! Liq from Dynamics
  REAL(KIND=JPRB)       :: PDYNI(KLON,KLEV,NBLOCKS) ! Liq from Dynamics
  REAL(KIND=JPRB)       :: PHRSW(KLON,KLEV,NBLOCKS) ! Short-wave heating rate
  REAL(KIND=JPRB)       :: PHRLW(KLON,KLEV,NBLOCKS) ! Long-wave heating rate
  REAL(KIND=JPRB)       :: PVERVEL(KLON,KLEV,NBLOCKS) !Vertical velocity
  REAL(KIND=JPRB)       :: PAP(KLON,KLEV,NBLOCKS)   ! Pressure on full levels
  REAL(KIND=JPRB)       :: PAPH(KLON,KLEV+1,NBLOCKS)! Pressure on half levels
  REAL(KIND=JPRB)       :: PLSM(KLON,NBLOCKS)       ! Land fraction (0-1) 
  LOGICAL               :: LDCUM(KLON,NBLOCKS)      ! Convection active
  INTEGER(KIND=JPIM)    :: KTYPE(KLON,NBLOCKS)      ! Convection type 0,1,2
  REAL(KIND=JPRB)       :: PLU(KLON,KLEV,NBLOCKS)   ! Conv. condensate
  REAL(KIND=JPRB)       :: PLUDE(KLON,KLEV,NBLOCKS) ! Conv. detrained water 
  REAL(KIND=JPRB)       :: PSNDE(KLON,KLEV,NBLOCKS) ! Conv. detrained snow
  REAL(KIND=JPRB)       :: PMFU(KLON,KLEV,NBLOCKS)  ! Conv. mass flux up
  REAL(KIND=JPRB)       :: PMFD(KLON,KLEV,NBLOCKS)  ! Conv. mass flux down
  REAL(KIND=JPRB)       :: PA(KLON,KLEV,NBLOCKS)    ! Original Cloud fraction (t)
  
  INTEGER(KIND=JPIM),PARAMETER    :: KFLDX =25
  
  
  REAL(KIND=JPRB)       :: PCLV(KLON,KLEV,NCLV,NBLOCKS) 
  
   ! Supersat clipped at previous time level in SLTEND
  REAL(KIND=JPRB)       :: PSUPSAT(KLON,KLEV,NBLOCKS)
  REAL(KIND=JPRB)      :: PCOVPTOT(KLON,KLEV,NBLOCKS) ! Precip fraction
  REAL(KIND=JPRB)      :: PRAINFRAC_TOPRFZ(KLON,NBLOCKS) 
  ! Flux diagnostics for DDH budget
  REAL(KIND=JPRB)      :: PFSQLF(KLON,KLEV+1,NBLOCKS)  ! Flux of liquid
  REAL(KIND=JPRB)      :: PFSQIF(KLON,KLEV+1,NBLOCKS)  ! Flux of ice
  REAL(KIND=JPRB)      :: PFCQLNG(KLON,KLEV+1,NBLOCKS) ! -ve corr for liq
  REAL(KIND=JPRB)      :: PFCQNNG(KLON,KLEV+1,NBLOCKS) ! -ve corr for ice
  REAL(KIND=JPRB)      :: PFSQRF(KLON,KLEV+1,NBLOCKS)  ! Flux diagnostics
  REAL(KIND=JPRB)      :: PFSQSF(KLON,KLEV+1,NBLOCKS)  !    for DDH, generic
  REAL(KIND=JPRB)      :: PFCQRNG(KLON,KLEV+1,NBLOCKS) ! rain
  REAL(KIND=JPRB)      :: PFCQSNG(KLON,KLEV+1,NBLOCKS) ! snow
  REAL(KIND=JPRB)      :: PFSQLTUR(KLON,KLEV+1,NBLOCKS) ! liquid flux due to VDF
  REAL(KIND=JPRB)      :: PFSQITUR(KLON,KLEV+1,NBLOCKS) ! ice flux due to VDF
  REAL(KIND=JPRB)      :: PFPLSL(KLON,KLEV+1,NBLOCKS) ! liq+rain sedim flux
  REAL(KIND=JPRB)      :: PFPLSN(KLON,KLEV+1,NBLOCKS) ! ice+snow sedim flux
  REAL(KIND=JPRB)      :: PFHPSL(KLON,KLEV+1,NBLOCKS) ! Enthalpy flux for liq
  REAL(KIND=JPRB)      :: PFHPSN(KLON,KLEV+1,NBLOCKS) ! Enthalp flux for ice
  LOGICAL              :: LDSLPHY 
  LOGICAL              :: LDMAINCALL       ! T if main call to cloudsc
  REAL(KIND=JPRB)      ::PEXTRA(KLON,KLEV,KFLDX,NBLOCKS) ! extra fields
  INTEGER(KIND=JPIM),PARAMETER    :: NPROMA =128
   INTEGER(KIND=JPIM),PARAMETER    :: NGPTOT =1024


  CALL CLOUDSCOUTER4(NBLOCKS, NGPTOT, NPROMA,    KLON,    KLEV,&
  & PTSPHY,&
  & PT, PQ,&
  & tendency_cml_T,&
  & tendency_cml_q,tendency_cml_a,&
  & tendency_cml_cld,&
  & tendency_tmp_T,&
  & tendency_tmp_q,tendency_tmp_a,&
  & tendency_tmp_cld,&
  & tendency_loc_T,&
  & tendency_loc_q,tendency_loc_a,&
  & tendency_loc_cld,&
  & PVFA, PVFL, PVFI, PDYNA, PDYNL, PDYNI, &
  & PHRSW,    PHRLW,&
  & PVERVEL,  PAP,      PAPH,&
  & PLSM,     LDCUM,    KTYPE, &
  & PLU,      PLUDE,    PSNDE,    PMFU,     PMFD,&
  & LDSLPHY,  LDMAINCALL, &
  !---prognostic fields
  & PA,&
  & PCLV,  &
  & PSUPSAT,&
!-- arrays for aerosol-cloud interactions
!!! & PQAER,    KAER, &
  & PLCRIT_AER,PICRIT_AER,&
  & PRE_ICE,&
  & PCCN,     PNICE,&
  !---diagnostic output
  & PCOVPTOT, PRAINFRAC_TOPRFZ,&
  !---resulting fluxes
  & PFSQLF,   PFSQIF ,  PFCQNNG,  PFCQLNG,&
  & PFSQRF,   PFSQSF ,  PFCQRNG,  PFCQSNG,&
  & PFSQLTUR, PFSQITUR , &
  & PFPLSL,   PFPLSN,   PFHPSL,   PFHPSN,&
  & PEXTRA,KFLDX)

end



SUBROUTINE CLOUDSCOUTER4&
  !---input
  & (NBLOCKS, NGPTOT, NPROMA,    KLON,    KLEV,&
  & PTSPHY,&
  & PT, PQ,&
  & tendency_cml_T,&
  & tendency_cml_q,tendency_cml_a,&
  & tendency_cml_cld,&
  & tendency_tmp_T,&
  & tendency_tmp_q,tendency_tmp_a,&
  & tendency_tmp_cld,&
  & tendency_loc_T,&
  & tendency_loc_q,tendency_loc_a,&
  & tendency_loc_cld,&
  & PVFA, PVFL, PVFI, PDYNA, PDYNL, PDYNI, &
  & PHRSW,    PHRLW,&
  & PVERVEL,  PAP,      PAPH,&
  & PLSM,     LDCUM,    KTYPE, &
  & PLU,      PLUDE,    PSNDE,    PMFU,     PMFD,&
  & LDSLPHY,  LDMAINCALL, &
  !---prognostic fields
  & PA,&
  & PCLV,  &
  & PSUPSAT,&
 !-- arrays for aerosol-cloud interactions
 !!! & PQAER,    KAER, &
  & PLCRIT_AER,PICRIT_AER,&
  & PRE_ICE,&
  & PCCN,     PNICE,&
  !---diagnostic output
  & PCOVPTOT, PRAINFRAC_TOPRFZ,&
  !---resulting fluxes
  & PFSQLF,   PFSQIF ,  PFCQNNG,  PFCQLNG,&
  & PFSQRF,   PFSQSF ,  PFCQRNG,  PFCQSNG,&
  & PFSQLTUR, PFSQITUR , &
  & PFPLSL,   PFPLSN,   PFHPSL,   PFHPSN,&
  & PEXTRA,KFLDX)
  USE PARKIND1 , ONLY : JPIM, JPRB
  USE YOMCST   , ONLY : RG, RD, RCPD, RETV, RLVTT, RLSTT, RLMLT, RTT, RV  
USE YOETHF   , ONLY : R2ES, R3LES, R3IES, R4LES, R4IES, R5LES, R5IES, &
 & R5ALVCP, R5ALSCP, RALVDCP, RALSDCP, RALFDCP, RTWAT, RTICE, RTICECU, &
 & RTWAT_RTICE_R, RTWAT_RTICECU_R, RKOOP1, RKOOP2
USE YOECLDP  , ONLY : NCLV, NCLDQL,NCLDQI,NCLDQR,NCLDQS, NCLDQV,RAMID,&
& RCLDIFF, RCLDIFF_CONVI, RCLCRIT,RCLCRIT_SEA, RCLCRIT_LAND,RKCONV,&
& RPRC1, RPRC2,RCLDMAX, RPECONS,RVRFACTOR, RPRECRHMAX,RTAUMEL, RAMIN,&
& RLMIN,RKOOPTAU, RCLDTOPP,RLCRITSNOW,RSNOWLIN1, RSNOWLIN2,RICEHI1,&
& RICEHI2, RICEINIT, RVICE,RVRAIN,RVSNOW,RTHOMO,RCOVPMIN, RCCN,RNICE,&
& RCCNOM, RCCNSS, RCCNSU, RCLDTOPCF, RDEPLIQREFRATE,RDEPLIQREFDEPTH,&
& RCL_KKAac, RCL_KKBac, RCL_KKAau,RCL_KKBauq, RCL_KKBaun,&
& RCL_KK_CLOUD_NUM_SEA, RCL_KK_CLOUD_NUM_LAND, RCL_AI, RCL_BI, RCL_CI,&
& RCL_DI, RCL_X1I, RCL_X2I, RCL_X3I,RCL_X4I, RCL_CONST1I, RCL_CONST2I,&
& RCL_CONST3I, RCL_CONST4I,RCL_CONST5I, RCL_CONST6I,RCL_APB1,RCL_APB2,&
& RCL_APB3, RCL_AS, RCL_BS,RCL_CS, RCL_DS,RCL_X1S, RCL_X2S,RCL_X3S,&
& RCL_X4S, RCL_CONST1S, RCL_CONST2S, RCL_CONST3S, RCL_CONST4S,&
& RCL_CONST5S, RCL_CONST6S,RCL_CONST7S,RCL_CONST8S,RDENSWAT, RDENSREF,&
& RCL_AR,RCL_BR, RCL_CR, RCL_DR, RCL_X1R, RCL_X2R, RCL_X4R,RCL_KA273,&
& RCL_CDENOM1, RCL_CDENOM2,RCL_CDENOM3,RCL_SCHMIDT,RCL_DYNVISC,RCL_CONST1R,&
& RCL_CONST2R, RCL_CONST3R, RCL_CONST4R, RCL_FAC1,RCL_FAC2, RCL_CONST5R,&
& RCL_CONST6R, RCL_FZRAB, RCL_FZRBB, LCLDEXTRA, LCLDBUDGET, NSSOPT, &
& NAECLBC, NAECLDU, NAECLOM, NAECLSS, NAECLSU, NCLDDIAG, NAERCLD, LAERLIQAUTOLSP,&
& LAERLIQAUTOCP, LAERLIQAUTOCPB,LAERLIQCOLL,LAERICESED,LAERICEAUTO, NSHAPEP,&
& NSHAPEQ,NBETA,RBETA,RBETAP1
!USE YOMJFH   , ONLY : N_VMASS
  IMPLICIT NONE
  SAVE
  INTEGER(KIND=JPIM)  :: KLON            ! Number of grid points
  INTEGER(KIND=JPIM)    :: KLEV           ! Number of levels
  INTEGER(KIND=JPIM)  :: NCLV            ! Number of levels
  INTEGER(KIND=JPIM)   :: KIDIA 
  INTEGER(KIND=JPIM)    :: KFDIA
  INTEGER(KIND=JPIM)   :: NBLOCKS 

  REAL(KIND=JPRB)       :: PLCRIT_AER(KLON,KLEV,NBLOCKS) 
  REAL(KIND=JPRB)       :: PICRIT_AER(KLON,KLEV,NBLOCKS) 
  REAL(KIND=JPRB)       :: PRE_ICE(KLON,KLEV,NBLOCKS) 
  REAL(KIND=JPRB)       :: PCCN(KLON,KLEV,NBLOCKS)     ! liquid cloud condensation nuclei
  REAL(KIND=JPRB)       :: PNICE(KLON,KLEV,NBLOCKS)    ! ice number concentration (cf. CCN)
  
  REAL(KIND=JPRB)       :: PTSPHY            ! Physics timestep
  REAL(KIND=JPRB)       :: PT(KLON,KLEV,NBLOCKS)    ! T at start of callpar
  REAL(KIND=JPRB)       :: PQ(KLON,KLEV,NBLOCKS)    ! Q at start of callpar
  
  REAL(KIND=JPRB) :: tendency_cml_T(KLON,KLEV,NBLOCKS)    ! GMV fields
  REAL(KIND=JPRB):: tendency_cml_q(KLON,KLEV,NBLOCKS) ,tendency_cml_a(KLON,KLEV,NBLOCKS)   ! GFL fields
  REAL(KIND=JPRB):: tendency_cml_cld(KLON,KLEV,NCLV,NBLOCKS)    ! composed cloud array
  
  REAL(KIND=JPRB):: tendency_tmp_T(KLON,KLEV,NBLOCKS)   ! GMV fields
  REAL(KIND=JPRB):: tendency_tmp_q(KLON,KLEV,NBLOCKS),tendency_tmp_a(KLON,KLEV,NBLOCKS)  ! GFL fields
  REAL(KIND=JPRB)  :: tendency_tmp_cld(KLON,KLEV,NCLV,NBLOCKS)   ! composed cloud array
  
  REAL(KIND=JPRB):: tendency_loc_T(KLON,KLEV,NBLOCKS)   ! GMV fields
  REAL(KIND=JPRB):: tendency_loc_q(KLON,KLEV,NBLOCKS),tendency_loc_a(KLON,KLEV,NBLOCKS)  ! GFL fields
  REAL(KIND=JPRB) :: tendency_loc_cld(KLON,KLEV,NCLV,NBLOCKS)   ! composed cloud array
  
  !TYPE (STATE_TYPE) , INTENT (IN)  :: tendency_cml   ! cumulative tendency used for final output
  !TYPE (STATE_TYPE) , INTENT (IN)  :: tendency_tmp   ! cumulative tendency used as input
  !TYPE (STATE_TYPE) , INTENT (OUT) :: tendency_loc   ! local tendency from cloud scheme
  REAL(KIND=JPRB)       :: PVFA(KLON,KLEV,NBLOCKS)  ! CC from VDF scheme
  REAL(KIND=JPRB)       :: PVFL(KLON,KLEV,NBLOCKS)  ! Liq from VDF scheme
  REAL(KIND=JPRB)       :: PVFI(KLON,KLEV,NBLOCKS)  ! Ice from VDF scheme
  REAL(KIND=JPRB)       :: PDYNA(KLON,KLEV,NBLOCKS) ! CC from Dynamics
  REAL(KIND=JPRB)       :: PDYNL(KLON,KLEV,NBLOCKS) ! Liq from Dynamics
  REAL(KIND=JPRB)       :: PDYNI(KLON,KLEV,NBLOCKS) ! Liq from Dynamics
  REAL(KIND=JPRB)       :: PHRSW(KLON,KLEV,NBLOCKS) ! Short-wave heating rate
  REAL(KIND=JPRB)       :: PHRLW(KLON,KLEV,NBLOCKS) ! Long-wave heating rate
  REAL(KIND=JPRB)       :: PVERVEL(KLON,KLEV,NBLOCKS) !Vertical velocity
  REAL(KIND=JPRB)       :: PAP(KLON,KLEV,NBLOCKS)   ! Pressure on full levels
  REAL(KIND=JPRB)       :: PAPH(KLON,KLEV+1,NBLOCKS)! Pressure on half levels
  REAL(KIND=JPRB)       :: PLSM(KLON,NBLOCKS)       ! Land fraction (0-1) 
  LOGICAL               :: LDCUM(KLON,NBLOCKS)      ! Convection active
  INTEGER(KIND=JPIM)    :: KTYPE(KLON,NBLOCKS)      ! Convection type 0,1,2
  REAL(KIND=JPRB)       :: PLU(KLON,KLEV,NBLOCKS)   ! Conv. condensate
  REAL(KIND=JPRB)    :: PLUDE(KLON,KLEV,NBLOCKS) ! Conv. detrained water 
  REAL(KIND=JPRB)       :: PSNDE(KLON,KLEV,NBLOCKS) ! Conv. detrained snow
  REAL(KIND=JPRB)       :: PMFU(KLON,KLEV,NBLOCKS)  ! Conv. mass flux up
  REAL(KIND=JPRB)       :: PMFD(KLON,KLEV,NBLOCKS)  ! Conv. mass flux down
  REAL(KIND=JPRB)       :: PA(KLON,KLEV,NBLOCKS)    ! Original Cloud fraction (t)
  
  INTEGER(KIND=JPIM)    :: KFLDX
  
  
  REAL(KIND=JPRB)       :: PCLV(KLON,KLEV,NCLV,NBLOCKS) 
  
   ! Supersat clipped at previous time level in SLTEND
  REAL(KIND=JPRB)       :: PSUPSAT(KLON,KLEV,NBLOCKS)
  REAL(KIND=JPRB)      :: PCOVPTOT(KLON,KLEV,NBLOCKS) ! Precip fraction
  REAL(KIND=JPRB)      :: PRAINFRAC_TOPRFZ(KLON,NBLOCKS) 
  ! Flux diagnostics for DDH budget
  REAL(KIND=JPRB)      :: PFSQLF(KLON,KLEV+1,NBLOCKS)  ! Flux of liquid
  REAL(KIND=JPRB)      :: PFSQIF(KLON,KLEV+1,NBLOCKS)  ! Flux of ice
  REAL(KIND=JPRB)      :: PFCQLNG(KLON,KLEV+1,NBLOCKS) ! -ve corr for liq
  REAL(KIND=JPRB)      :: PFCQNNG(KLON,KLEV+1,NBLOCKS) ! -ve corr for ice
  REAL(KIND=JPRB)      :: PFSQRF(KLON,KLEV+1,NBLOCKS)  ! Flux diagnostics
  REAL(KIND=JPRB)      :: PFSQSF(KLON,KLEV+1,NBLOCKS)  !    for DDH, generic
  REAL(KIND=JPRB)      :: PFCQRNG(KLON,KLEV+1,NBLOCKS) ! rain
  REAL(KIND=JPRB)      :: PFCQSNG(KLON,KLEV+1,NBLOCKS) ! snow
  REAL(KIND=JPRB)      :: PFSQLTUR(KLON,KLEV+1,NBLOCKS) ! liquid flux due to VDF
  REAL(KIND=JPRB)      :: PFSQITUR(KLON,KLEV+1,NBLOCKS) ! ice flux due to VDF
  REAL(KIND=JPRB)      :: PFPLSL(KLON,KLEV+1,NBLOCKS) ! liq+rain sedim flux
  REAL(KIND=JPRB)      :: PFPLSN(KLON,KLEV+1,NBLOCKS) ! ice+snow sedim flux
  REAL(KIND=JPRB)      :: PFHPSL(KLON,KLEV+1,NBLOCKS) ! Enthalpy flux for liq
  REAL(KIND=JPRB)      :: PFHPSN(KLON,KLEV+1,NBLOCKS) ! Enthalp flux for ice
  LOGICAL              :: LDSLPHY 
  LOGICAL              :: LDMAINCALL       ! T if main call to cloudsc
  REAL(KIND=JPRB)      ::PEXTRA(KLON,KLEV,KFLDX,NBLOCKS) ! extra fields
  INTEGER(KIND=JPIM) :: IBL
  INTEGER(KIND=JPIM),PARAMETER  :: NGPTOT,NPROMA 

  DO IBL=1,NBLOCKS
      !JKGLO=(IBL-1)*NPROMA+1

      
      !ICEND=MIN(NPROMA,NGPTOT-JKGLO+1)
      !ICEND=MIN(128,65536-JKGLO+1)
      !ICEND=128
      !-- These were uninitialized : meaningful only when we compare error differences
      !PCOVPTOT(:,:,IBL) = 0.0
      !tendency_loc_cld(:,:,NCLV,IBL) = 0.0

      CALL CLOUDSC(1,    NPROMA,    KLON,    KLEV,&
      & PTSPHY,&
      & PT(:,:,IBL), PQ(:,:,IBL),&
      & tendency_cml_T(:,:,IBL),&
      & tendency_cml_q(:,:,IBL),tendency_cml_a(:,:,IBL),&
      & tendency_cml_cld(:,:,:,IBL),&
      & tendency_tmp_T(:,:,IBL),&
      & tendency_tmp_q(:,:,IBL),tendency_tmp_a(:,:,IBL),&
      & tendency_tmp_cld(:,:,:,IBL),&
      & tendency_loc_T(:,:,IBL),&
      & tendency_loc_q(:,:,IBL),tendency_loc_a(:,:,IBL),&
      & tendency_loc_cld(:,:,:,IBL),&
      & PVFA(:,:,IBL), PVFL(:,:,IBL), PVFI(:,:,IBL), PDYNA(:,:,IBL), PDYNL(:,:,IBL), PDYNI(:,:,IBL), &
      & PHRSW(:,:,IBL),    PHRLW(:,:,IBL),&
      & PVERVEL(:,:,IBL),  PAP(:,:,IBL),      PAPH(:,:,IBL),&
      & PLSM(:,IBL),     LDCUM(:,IBL),    KTYPE(:,IBL), &
      & PLU(:,:,IBL),      PLUDE(:,:,IBL),    PSNDE(:,:,IBL),    PMFU(:,:,IBL),     PMFD(:,:,IBL),&
      & LDSLPHY,  LDMAINCALL, &
      !---prognostic fields
      & PA(:,:,IBL),&
      & PCLV(:,:,:,IBL),  &
      & PSUPSAT(:,:,IBL),&
    !-- arrays for aerosol-cloud interactions
    !!! & PQAER,    KAER, &
      & PLCRIT_AER(:,:,IBL),PICRIT_AER(:,:,IBL),&
      & PRE_ICE(:,:,IBL),&
      & PCCN(:,:,IBL),     PNICE(:,:,IBL),&
      !---diagnostic output
      & PCOVPTOT(:,:,IBL), PRAINFRAC_TOPRFZ(:,IBL),&
      !---resulting fluxes
      & PFSQLF(:,:,IBL),   PFSQIF(:,:,IBL) ,  PFCQNNG(:,:,IBL),  PFCQLNG(:,:,IBL),&
      & PFSQRF(:,:,IBL),   PFSQSF(:,:,IBL) ,  PFCQRNG(:,:,IBL),  PFCQSNG(:,:,IBL),&
      & PFSQLTUR(:,:,IBL), PFSQITUR(:,:,IBL) , &
      & PFPLSL(:,:,IBL),   PFPLSN(:,:,IBL),   PFHPSL(:,:,IBL),   PFHPSN(:,:,IBL),&
      & PEXTRA(:,:,:,IBL),KFLDX,&
      &RG, RD, RCPD, RETV, RLVTT, RLSTT, RLMLT, RTT, RV,&  
&R2ES, R3LES, R3IES, R4LES, R4IES, R5LES, R5IES, &
 & R5ALVCP, R5ALSCP, RALVDCP, RALSDCP, RALFDCP, RTWAT, RTICE, RTICECU, &
 & RTWAT_RTICE_R, RTWAT_RTICECU_R, RKOOP1, RKOOP2,&
& NCLV, NCLDQL,NCLDQI,NCLDQR,NCLDQS, NCLDQV,RAMID,&
& RCLDIFF, RCLDIFF_CONVI, RCLCRIT_SEA, RCLCRIT_LAND,RKCONV,&
& RPRC1, RPECONS,RVRFACTOR, RPRECRHMAX,RTAUMEL, RAMIN,&
& RLMIN,RKOOPTAU, RLCRITSNOW,RSNOWLIN1, RSNOWLIN2,&
& RICEINIT, RVICE,RVRAIN,RVSNOW,RTHOMO,RCOVPMIN, RCCN,RNICE,&
& RCLDTOPCF, RDEPLIQREFRATE,RDEPLIQREFDEPTH,&
& RCL_KKAac, RCL_KKBac, RCL_KKAau,RCL_KKBauq, RCL_KKBaun,&
& RCL_KK_CLOUD_NUM_SEA, RCL_KK_CLOUD_NUM_LAND,&
& RCL_CONST1I, RCL_CONST2I,&
& RCL_CONST3I, RCL_CONST4I,RCL_CONST5I, RCL_CONST6I,RCL_APB1,RCL_APB2,&
& RCL_APB3,&
& RCL_CONST1S, RCL_CONST2S, RCL_CONST3S, RCL_CONST4S,&
& RCL_CONST5S, RCL_CONST6S,RCL_CONST7S,RCL_CONST8S, RDENSREF,&
& RCL_KA273,&
& RCL_CDENOM1, RCL_CDENOM2,RCL_CDENOM3,RCL_CONST1R,&
& RCL_CONST2R, RCL_CONST3R, RCL_CONST4R, RCL_FAC1,RCL_FAC2, RCL_CONST5R,&
& RCL_CONST6R, RCL_FZRAB,  NSSOPT, &
&  LAERLIQAUTOLSP,&
& LAERLIQCOLL,LAERICESED,LAERICEAUTO)
  ENDDO

end


SUBROUTINE CLOUDSC&
 !---input
 & (KIDIA,    KFDIA,    KLON,    KLEV,&
 & PTSPHY,&
 & PT, PQ,&
 & tendency_cml_T,&
 & tendency_cml_q,tendency_cml_a,&
 & tendency_cml_cld,&
 & tendency_tmp_T,&
 & tendency_tmp_q,tendency_tmp_a,&
 & tendency_tmp_cld,&
 & tendency_loc_T,&
 & tendency_loc_q,tendency_loc_a,&
 & tendency_loc_cld,&
 & PVFA, PVFL, PVFI, PDYNA, PDYNL, PDYNI, &
 & PHRSW,    PHRLW,&
 & PVERVEL,  PAP,      PAPH,&
 & PLSM,     LDCUM,    KTYPE, &
 & PLU,      PLUDE,    PSNDE,    PMFU,     PMFD,&
 & LDSLPHY,  LDMAINCALL, &
 !---prognostic fields
 & PA,&
 & PCLV,  &
 & PSUPSAT,&
!-- arrays for aerosol-cloud interactions
!!! & PQAER,    KAER, &
 & PLCRIT_AER,PICRIT_AER,&
 & PRE_ICE,&
 & PCCN,     PNICE,&
 !---diagnostic output
 & PCOVPTOT, PRAINFRAC_TOPRFZ,&
 !---resulting fluxes
 & PFSQLF,   PFSQIF ,  PFCQNNG,  PFCQLNG,&
 & PFSQRF,   PFSQSF ,  PFCQRNG,  PFCQSNG,&
 & PFSQLTUR, PFSQITUR , &
 & PFPLSL,   PFPLSN,   PFHPSL,   PFHPSN,&
 & PEXTRA,KFLDX,&
 &RG, RD, RCPD, RETV, RLVTT, RLSTT, RLMLT, RTT, RV,&  
&R2ES, R3LES, R3IES, R4LES, R4IES, R5LES, R5IES, &
 & R5ALVCP, R5ALSCP, RALVDCP, RALSDCP, RALFDCP, RTWAT, RTICE, RTICECU, &
 & RTWAT_RTICE_R, RTWAT_RTICECU_R, RKOOP1, RKOOP2,&
& NCLV, NCLDQL,NCLDQI,NCLDQR,NCLDQS, NCLDQV,RAMID,&
& RCLDIFF, RCLDIFF_CONVI, RCLCRIT_SEA, RCLCRIT_LAND,RKCONV,&
& RPRC1, RPECONS,RVRFACTOR, RPRECRHMAX,RTAUMEL, RAMIN,&
& RLMIN,RKOOPTAU, RLCRITSNOW,RSNOWLIN1, RSNOWLIN2,&
& RICEINIT, RVICE,RVRAIN,RVSNOW,RTHOMO,RCOVPMIN, RCCN,RNICE,&
& RCLDTOPCF, RDEPLIQREFRATE,RDEPLIQREFDEPTH,&
& RCL_KKAac, RCL_KKBac, RCL_KKAau,RCL_KKBauq, RCL_KKBaun,&
& RCL_KK_CLOUD_NUM_SEA, RCL_KK_CLOUD_NUM_LAND,&
& RCL_CONST1I, RCL_CONST2I,&
& RCL_CONST3I, RCL_CONST4I,RCL_CONST5I, RCL_CONST6I,RCL_APB1,RCL_APB2,&
& RCL_APB3,&
& RCL_CONST1S, RCL_CONST2S, RCL_CONST3S, RCL_CONST4S,&
& RCL_CONST5S, RCL_CONST6S,RCL_CONST7S,RCL_CONST8S, RDENSREF,&
& RCL_KA273,&
& RCL_CDENOM1, RCL_CDENOM2,RCL_CDENOM3,RCL_CONST1R,&
& RCL_CONST2R, RCL_CONST3R, RCL_CONST4R, RCL_FAC1,RCL_FAC2, RCL_CONST5R,&
& RCL_CONST6R, RCL_FZRAB,  NSSOPT, &
&  LAERLIQAUTOLSP,&
& LAERLIQCOLL,LAERICESED,LAERICEAUTO)

!===============================================================================
!**** *CLOUDSC* -  ROUTINE FOR PARAMATERIZATION OF CLOUD PROCESSES
!                  FOR PROGNOSTIC CLOUD SCHEME
!!
!     M.Tiedtke, C.Jakob, A.Tompkins, R.Forbes     (E.C.M.W.F.)
!!
!     PURPOSE
!     -------
!          THIS ROUTINE UPDATES THE CONV/STRAT CLOUD FIELDS.
!          THE FOLLOWING PROCESSES ARE CONSIDERED:
!        - Detrainment of cloud water from convective updrafts
!        - Evaporation/condensation of cloud water in connection
!           with heating/cooling such as by subsidence/ascent
!        - Erosion of clouds by turbulent mixing of cloud air
!           with unsaturated environmental air
!        - Deposition onto ice when liquid water present (Bergeron-Findeison) 
!        - Conversion of cloud water into rain (collision-coalescence)
!        - Conversion of cloud ice to snow (aggregation)
!        - Sedimentation of rain, snow and ice
!        - Evaporation of rain and snow
!        - Melting of snow and ice
!        - Freezing of liquid and rain
!        Note: Turbulent transports of s,q,u,v at cloud tops due to
!           buoyancy fluxes and lw radiative cooling are treated in 
!           the VDF scheme
!!
!     INTERFACE.
!     ----------
!          *CLOUDSC* IS CALLED FROM *CALLPAR*
!     THE ROUTINE TAKES ITS INPUT FROM THE LONG-TERM STORAGE:
!     T,Q,L,PHI AND DETRAINMENT OF CLOUD WATER FROM THE
!     CONVECTIVE CLOUDS (MASSFLUX CONVECTION SCHEME), BOUNDARY
!     LAYER TURBULENT FLUXES OF HEAT AND MOISTURE, RADIATIVE FLUXES,
!     OMEGA.
!     IT RETURNS ITS OUTPUT TO:
!      1.MODIFIED TENDENCIES OF MODEL VARIABLES T AND Q
!        AS WELL AS CLOUD VARIABLES L AND C
!      2.GENERATES PRECIPITATION FLUXES FROM STRATIFORM CLOUDS
!!
!     EXTERNALS.
!     ----------
!          NONE
!!
!     MODIFICATIONS.
!     -------------
!      M. TIEDTKE    E.C.M.W.F.     8/1988, 2/1990
!     CH. JAKOB      E.C.M.W.F.     2/1994 IMPLEMENTATION INTO IFS
!     A.TOMPKINS     E.C.M.W.F.     2002   NEW NUMERICS
!        01-05-22 : D.Salmond   Safety modifications
!        02-05-29 : D.Salmond   Optimisation
!        03-01-13 : J.Hague     MASS Vector Functions  J.Hague
!        03-10-01 : M.Hamrud    Cleaning
!        04-12-14 : A.Tompkins  New implicit solver and physics changes
!        04-12-03 : A.Tompkins & M.Ko"hler  moist PBL
!     G.Mozdzynski  09-Jan-2006  EXP security fix
!        19-01-09 : P.Bechtold  Changed increased RCLDIFF value for KTYPE=2
!        07-07-10 : A.Tompkins/R.Forbes  4-Phase flexible microphysics
!        01-03-11 : R.Forbes    Mixed phase changes and tidy up
!        01-10-11 : R.Forbes    Melt ice to rain, allow rain to freeze
!        01-10-11 : R.Forbes    Limit supersat to avoid excessive values
!        31-10-11 : M.Ahlgrimm  Add rain, snow and PEXTRA to DDH output
!        17-02-12 : F.Vana      Simplified/optimized LU factorization
!        18-05-12 : F.Vana      Cleaning + better support of sequential physics
!        N.Semane+P.Bechtold     04-10-2012 Add RVRFACTOR factor for small planet
!        01-02-13 : R.Forbes    New params of autoconv/acc,rain evap,snow riming
!        15-03-13 : F. Vana     New dataflow + more tendencies from the first call
!        K. Yessad (July 2014): Move some variables.
!        F. Vana  05-Mar-2015  Support for single precision
!        15-01-15 : R.Forbes    Added new options for snow evap & ice deposition
!        10-01-15 : R.Forbes    New physics for rain freezing
!        23-10-14 : P. Bechtold remove zeroing of convection arrays
!
!     SWITCHES.
!     --------
!!
!     MODEL PARAMETERS
!     ----------------
!     RCLDIFF:    PARAMETER FOR EROSION OF CLOUDS
!     RCLCRIT_SEA:  THRESHOLD VALUE FOR RAIN AUTOCONVERSION OVER SEA
!     RCLCRIT_LAND: THRESHOLD VALUE FOR RAIN AUTOCONVERSION OVER LAND
!     RLCRITSNOW: THRESHOLD VALUE FOR SNOW AUTOCONVERSION
!     RKCONV:     PARAMETER FOR AUTOCONVERSION OF CLOUDS (KESSLER)
!     RCLDMAX:    MAXIMUM POSSIBLE CLW CONTENT (MASON,1971)
!!
!     REFERENCES.
!     ----------
!     TIEDTKE MWR 1993
!     JAKOB PhD 2000
!     GREGORY ET AL. QJRMS 2000
!     TOMPKINS ET AL. QJRMS 2007
!!
!===============================================================================

USE PARKIND1 , ONLY : JPIM, JPRB
!USE YOMHOOK  , ONLY : LHOOK, DR_HOOK


IMPLICIT NONE

!-------------------------------------------------------------------------------
!                 Declare input/output arguments
!-------------------------------------------------------------------------------
 
! PLCRIT_AER : critical liquid mmr for rain autoconversion process
! PICRIT_AER : critical liquid mmr for snow autoconversion process
! PRE_LIQ : liq Re
! PRE_ICE : ice Re
! PCCN    : liquid cloud condensation nuclei
! PNICE   : ice number concentration (cf. CCN)
REAL(KIND=JPRB) :: R2ES
REAL(KIND=JPRB) :: R3IES
REAL(KIND=JPRB) :: R3LES
REAL(KIND=JPRB) :: R5ALSCP
REAL(KIND=JPRB) :: R5ALVCP
REAL(KIND=JPRB) :: RTICE
REAL(KIND=JPRB) :: RTICECU
REAL(KIND=JPRB) :: RTWAT_RTICE_R
REAL(KIND=JPRB) :: RTWAT_RTICECU_R
REAL(KIND=JPRB) :: RTWAT
REAL(KIND=JPRB) :: RKOOP1
REAL(KIND=JPRB) :: RKOOP2
REAL(KIND=JPRB) :: RCL_KK_CLOUD_NUM_SEA
REAL(KIND=JPRB) :: RCL_KK_CLOUD_NUM_LAND
REAL(KIND=JPRB) :: RCL_APB2
REAL(KIND=JPRB) :: RCL_APB3
REAL(KIND=JPRB) :: RCLDTOPCF
REAL(KIND=JPRB) :: RALFDCP
REAL(KIND=JPRB) :: RCL_CONST2S
REAL(KIND=JPRB) :: RCL_CONST6S
REAL(KIND=JPRB) :: RCL_KKBauq
REAL(KIND=JPRB) :: RSNOWLIN1
REAL(KIND=JPRB) :: RPRC1
REAL(KIND=JPRB) :: RKCONV
REAL(KIND=JPRB) :: RVSNOW
REAL(KIND=JPRB) :: RCL_CONST5S
REAL(KIND=JPRB) :: RCL_KKBac
REAL(KIND=JPRB) :: RETV
REAL(KIND=JPRB) :: RCL_CONST2I
REAL(KIND=JPRB) :: RCLCRIT_LAND
REAL(KIND=JPRB) :: RCL_CONST3S
REAL(KIND=JPRB) :: RCL_CONST7S
REAL(KIND=JPRB) :: RSNOWLIN2
REAL(KIND=JPRB) :: RCL_FAC2
REAL(KIND=JPRB) :: RTHOMO
REAL(KIND=JPRB) :: RCOVPMIN
REAL(KIND=JPRB) :: RCL_FZRAB
LOGICAL :: LAERLIQCOLL
REAL(KIND=JPRB) :: RCL_CDENOM2
REAL(KIND=JPRB) :: RPECONS
REAL(KIND=JPRB) :: RCL_CONST1R
REAL(KIND=JPRB) :: RDENSREF
REAL(KIND=JPRB) :: RNICE
INTEGER(KIND=JPIM) :: NSSOPT
REAL(KIND=JPRB) :: RVICE
REAL(KIND=JPRB) :: RCLDIFF
REAL(KIND=JPRB) :: RAMIN
REAL(KIND=JPRB) :: RCL_CONST4I
REAL(KIND=JPRB) :: RTT
LOGICAL :: LAERICEAUTO
REAL(KIND=JPRB) :: RCL_CONST3R
REAL(KIND=JPRB) :: RLVTT
REAL(KIND=JPRB) :: RV
REAL(KIND=JPRB) :: RLSTT
REAL(KIND=JPRB) :: RCL_CONST5I
REAL(KIND=JPRB) :: RVRAIN
REAL(KIND=JPRB) :: RCL_CONST2R
REAL(KIND=JPRB) :: RCL_KA273
REAL(KIND=JPRB) :: RCL_CONST6R
REAL(KIND=JPRB) :: RTAUMEL
REAL(KIND=JPRB) :: RICEINIT
REAL(KIND=JPRB) :: RDEPLIQREFDEPTH
REAL(KIND=JPRB) :: R4LES
REAL(KIND=JPRB) :: RCL_CONST1S
REAL(KIND=JPRB) :: RCL_CONST4R
REAL(KIND=JPRB) :: RPRECRHMAX
REAL(KIND=JPRB) :: RCL_CONST1I
REAL(KIND=JPRB) :: RKOOPTAU
REAL(KIND=JPRB) :: RLCRITSNOW
REAL(KIND=JPRB) :: R5LES
LOGICAL :: LAERICESED
REAL(KIND=JPRB) :: RCL_CONST3I
REAL(KIND=JPRB) :: RCLCRIT_SEA
REAL(KIND=JPRB) :: RAMID
REAL(KIND=JPRB) :: RCL_KKAau
REAL(KIND=JPRB) :: R4IES
REAL(KIND=JPRB) :: RCL_APB1
REAL(KIND=JPRB) :: RCL_CONST6I
REAL(KIND=JPRB) :: RDEPLIQREFRATE
REAL(KIND=JPRB) :: RLMIN
REAL(KIND=JPRB) :: RCL_FAC1
REAL(KIND=JPRB) :: RCL_KKBaun
REAL(KIND=JPRB) :: RALVDCP
REAL(KIND=JPRB) :: RVRFACTOR
REAL(KIND=JPRB) :: RALSDCP
LOGICAL :: LAERLIQAUTOLSP
REAL(KIND=JPRB) :: RCL_CONST8S
REAL(KIND=JPRB) :: RCL_CONST5R
REAL(KIND=JPRB) :: RCCN
REAL(KIND=JPRB) :: RCLDIFF_CONVI
REAL(KIND=JPRB) :: RD
REAL(KIND=JPRB) :: R5IES
REAL(KIND=JPRB) :: RCL_CONST4S
REAL(KIND=JPRB) :: RCPD
REAL(KIND=JPRB) :: RLMLT
REAL(KIND=JPRB) :: RG
REAL(KIND=JPRB) :: RCL_CDENOM1
REAL(KIND=JPRB) :: RCL_CDENOM3
REAL(KIND=JPRB) :: RCL_KKAac
INTEGER(KIND=JPIM) :: NCLV
INTEGER(KIND=JPIM) :: NCLDQL
INTEGER(KIND=JPIM) :: NCLDQI
INTEGER(KIND=JPIM) :: NCLDQR
INTEGER(KIND=JPIM) :: NCLDQS
INTEGER(KIND=JPIM) :: NCLDQV

INTEGER(KIND=JPIM) :: IBL,ICEND, NGPTOT, NPROMA
REAL(KIND=JPRB)   ,INTENT(IN)    :: PLCRIT_AER(KLON,KLEV) 
REAL(KIND=JPRB)   ,INTENT(IN)    :: PICRIT_AER(KLON,KLEV) 
REAL(KIND=JPRB)   ,INTENT(IN)    :: PRE_ICE(KLON,KLEV) 
REAL(KIND=JPRB)   ,INTENT(IN)    :: PCCN(KLON,KLEV)     ! liquid cloud condensation nuclei
REAL(KIND=JPRB)   ,INTENT(IN)    :: PNICE(KLON,KLEV)    ! ice number concentration (cf. CCN)

INTEGER(KIND=JPIM),INTENT(IN)    :: KLON             ! Number of grid points
INTEGER(KIND=JPIM),INTENT(IN)    :: KLEV             ! Number of levels
INTEGER(KIND=JPIM),INTENT(IN)    :: KIDIA 
INTEGER(KIND=JPIM),INTENT(IN)    :: KFDIA 
REAL(KIND=JPRB)   ,INTENT(IN)    :: PTSPHY            ! Physics timestep
REAL(KIND=JPRB)   ,INTENT(IN)    :: PT(KLON,KLEV)    ! T at start of callpar
REAL(KIND=JPRB)   ,INTENT(IN)    :: PQ(KLON,KLEV)    ! Q at start of callpar
LOGICAL           ,INTENT(IN)    :: LDSLPHY 
LOGICAL           ,INTENT(IN)    :: LDMAINCALL       ! T if main call to cloudsc
REAL(KIND=JPRB)   ,INTENT(INOUT) :: PEXTRA(KLON,KLEV,KFLDX) ! extra fields
REAL(KIND=JPRB) :: tendency_cml_T(KLON,KLEV)    ! GMV fields
REAL(KIND=JPRB):: tendency_cml_q(KLON,KLEV) ,tendency_cml_a(KLON,KLEV)   ! GFL fields
REAL(KIND=JPRB):: tendency_cml_cld(KLON,KLEV,NCLV)    ! composed cloud array

REAL(KIND=JPRB):: tendency_tmp_T(KLON,KLEV)   ! GMV fields
REAL(KIND=JPRB):: tendency_tmp_q(KLON,KLEV),tendency_tmp_a(KLON,KLEV)  ! GFL fields
REAL(KIND=JPRB)  :: tendency_tmp_cld(KLON,KLEV,NCLV)   ! composed cloud array

REAL(KIND=JPRB):: tendency_loc_T(KLON,KLEV)   ! GMV fields
REAL(KIND=JPRB):: tendency_loc_q(KLON,KLEV),tendency_loc_a(KLON,KLEV)  ! GFL fields
REAL(KIND=JPRB) :: tendency_loc_cld(KLON,KLEV,NCLV)   ! composed cloud array

!TYPE (STATE_TYPE) , INTENT (IN)  :: tendency_cml   ! cumulative tendency used for final output
!TYPE (STATE_TYPE) , INTENT (IN)  :: tendency_tmp   ! cumulative tendency used as input
!TYPE (STATE_TYPE) , INTENT (OUT) :: tendency_loc   ! local tendency from cloud scheme
REAL(KIND=JPRB)   ,INTENT(IN)    :: PVFA(KLON,KLEV)  ! CC from VDF scheme
REAL(KIND=JPRB)   ,INTENT(IN)    :: PVFL(KLON,KLEV)  ! Liq from VDF scheme
REAL(KIND=JPRB)   ,INTENT(IN)    :: PVFI(KLON,KLEV)  ! Ice from VDF scheme
REAL(KIND=JPRB)   ,INTENT(IN)    :: PDYNA(KLON,KLEV) ! CC from Dynamics
REAL(KIND=JPRB)   ,INTENT(IN)    :: PDYNL(KLON,KLEV) ! Liq from Dynamics
REAL(KIND=JPRB)   ,INTENT(IN)    :: PDYNI(KLON,KLEV) ! Liq from Dynamics
REAL(KIND=JPRB)   ,INTENT(IN)    :: PHRSW(KLON,KLEV) ! Short-wave heating rate
REAL(KIND=JPRB)   ,INTENT(IN)    :: PHRLW(KLON,KLEV) ! Long-wave heating rate
REAL(KIND=JPRB)   ,INTENT(IN)    :: PVERVEL(KLON,KLEV) !Vertical velocity
REAL(KIND=JPRB)   ,INTENT(IN)    :: PAP(KLON,KLEV)   ! Pressure on full levels
REAL(KIND=JPRB)   ,INTENT(IN)    :: PAPH(KLON,KLEV+1)! Pressure on half levels
REAL(KIND=JPRB)   ,INTENT(IN)    :: PLSM(KLON)       ! Land fraction (0-1) 
LOGICAL           ,INTENT(IN)    :: LDCUM(KLON)      ! Convection active
INTEGER(KIND=JPIM),INTENT(IN)    :: KTYPE(KLON)      ! Convection type 0,1,2
REAL(KIND=JPRB)   ,INTENT(IN)    :: PLU(KLON,KLEV)   ! Conv. condensate
REAL(KIND=JPRB)   ,INTENT(INOUT) :: PLUDE(KLON,KLEV) ! Conv. detrained water 
REAL(KIND=JPRB)   ,INTENT(IN)    :: PSNDE(KLON,KLEV) ! Conv. detrained snow
REAL(KIND=JPRB)   ,INTENT(IN)    :: PMFU(KLON,KLEV)  ! Conv. mass flux up
REAL(KIND=JPRB)   ,INTENT(IN)    :: PMFD(KLON,KLEV)  ! Conv. mass flux down
REAL(KIND=JPRB)   ,INTENT(IN)    :: PA(KLON,KLEV)    ! Original Cloud fraction (t)

INTEGER(KIND=JPIM),INTENT(IN)    :: KFLDX 

REAL(KIND=JPRB)   ,INTENT(IN)    :: PCLV(KLON,KLEV,NCLV) 

 ! Supersat clipped at previous time level in SLTEND
REAL(KIND=JPRB)   ,INTENT(IN)    :: PSUPSAT(KLON,KLEV)
REAL(KIND=JPRB)   ,INTENT(OUT)   :: PCOVPTOT(KLON,KLEV) ! Precip fraction
REAL(KIND=JPRB)   ,INTENT(OUT)   :: PRAINFRAC_TOPRFZ(KLON) 
! Flux diagnostics for DDH budget
REAL(KIND=JPRB)   ,INTENT(OUT)   :: PFSQLF(KLON,KLEV+1)  ! Flux of liquid
REAL(KIND=JPRB)   ,INTENT(OUT)   :: PFSQIF(KLON,KLEV+1)  ! Flux of ice
REAL(KIND=JPRB)   ,INTENT(OUT)   :: PFCQLNG(KLON,KLEV+1) ! -ve corr for liq
REAL(KIND=JPRB)   ,INTENT(OUT)   :: PFCQNNG(KLON,KLEV+1) ! -ve corr for ice
REAL(KIND=JPRB)   ,INTENT(OUT)   :: PFSQRF(KLON,KLEV+1)  ! Flux diagnostics
REAL(KIND=JPRB)   ,INTENT(OUT)   :: PFSQSF(KLON,KLEV+1)  !    for DDH, generic
REAL(KIND=JPRB)   ,INTENT(OUT)   :: PFCQRNG(KLON,KLEV+1) ! rain
REAL(KIND=JPRB)   ,INTENT(OUT)   :: PFCQSNG(KLON,KLEV+1) ! snow
REAL(KIND=JPRB)   ,INTENT(OUT)   :: PFSQLTUR(KLON,KLEV+1) ! liquid flux due to VDF
REAL(KIND=JPRB)   ,INTENT(OUT)   :: PFSQITUR(KLON,KLEV+1) ! ice flux due to VDF
REAL(KIND=JPRB)   ,INTENT(OUT)   :: PFPLSL(KLON,KLEV+1) ! liq+rain sedim flux
REAL(KIND=JPRB)   ,INTENT(OUT)   :: PFPLSN(KLON,KLEV+1) ! ice+snow sedim flux
REAL(KIND=JPRB)   ,INTENT(OUT)   :: PFHPSL(KLON,KLEV+1) ! Enthalpy flux for liq
REAL(KIND=JPRB)   ,INTENT(OUT)   :: PFHPSN(KLON,KLEV+1) ! Enthalp flux for ice


!-------------------------------------------------------------------------------
!                       Declare local variables
!-------------------------------------------------------------------------------

REAL(KIND=JPRB) :: &
!  condensation and evaporation terms
 & ZLCOND1(KLON), ZLCOND2(KLON),&
 & ZLEVAP,        ZLEROS,&
 & ZLEVAPL(KLON), ZLEVAPI(KLON),&
! autoconversion terms
 & ZRAINAUT(KLON), ZSNOWAUT(KLON), &
 & ZLIQCLD(KLON),  ZICECLD(KLON)
REAL(KIND=JPRB) :: ZFOKOOP(KLON), ZFOEALFA(KLON,KLEV+1)
REAL(KIND=JPRB) :: ZICENUCLEI(KLON) ! number concentration of ice nuclei

REAL(KIND=JPRB) :: ZLICLD(KLON)
REAL(KIND=JPRB) :: ZACOND
REAL(KIND=JPRB) :: ZAEROS
REAL(KIND=JPRB) :: ZLFINALSUM(KLON)
REAL(KIND=JPRB) :: ZDQS(KLON)
REAL(KIND=JPRB) :: ZTOLD(KLON)
REAL(KIND=JPRB) :: ZQOLD(KLON)  
REAL(KIND=JPRB) :: ZDTGDP(KLON) 
REAL(KIND=JPRB) :: ZRDTGDP(KLON)  
REAL(KIND=JPRB) :: ZCOVPCLR(KLON)   
REAL(KIND=JPRB) :: ZPRECLR
REAL(KIND=JPRB) :: ZCOVPTOT(KLON)    
REAL(KIND=JPRB) :: ZCOVPMAX(KLON)
REAL(KIND=JPRB) :: ZQPRETOT(KLON)
REAL(KIND=JPRB) :: ZDPEVAP
REAL(KIND=JPRB) :: ZDTFORC
REAL(KIND=JPRB) :: ZDTDIAB
REAL(KIND=JPRB) :: ZTP1(KLON,KLEV)   
REAL(KIND=JPRB) :: ZLDEFR(KLON)
REAL(KIND=JPRB) :: ZLDIFDT(KLON)
REAL(KIND=JPRB) :: ZDTGDPF(KLON)
REAL(KIND=JPRB) :: ZLCUST(KLON,NCLV)
REAL(KIND=JPRB) :: ZACUST(KLON)
REAL(KIND=JPRB) :: ZMF(KLON) 

REAL(KIND=JPRB) :: ZRHO(KLON)
REAL(KIND=JPRB) :: ZTMP1(KLON),ZTMP2(KLON),ZTMP3(KLON)
REAL(KIND=JPRB) :: ZTMP4(KLON),ZTMP5(KLON),ZTMP6(KLON),ZTMP7(KLON)
REAL(KIND=JPRB) :: ZALFAWM(KLON)

! Accumulators of A,B,and C factors for cloud equations
REAL(KIND=JPRB) :: ZSOLAB(KLON) ! -ve implicit CC
REAL(KIND=JPRB) :: ZSOLAC(KLON) ! linear CC
REAL(KIND=JPRB) :: ZANEW
REAL(KIND=JPRB) :: ZANEWM1(KLON) 

REAL(KIND=JPRB) :: ZGDP(KLON)

!---for flux calculation
REAL(KIND=JPRB) :: ZDA(KLON)
REAL(KIND=JPRB) :: ZLI(KLON,KLEV),           ZA(KLON,KLEV)
REAL(KIND=JPRB) :: ZAORIG(KLON,KLEV) ! start of scheme value for CC

LOGICAL :: LLFLAG(KLON)
LOGICAL :: LLO1

INTEGER(KIND=JPIM) :: ICALL, IK, JK, JL, JM, JN, JO, JLEN, IS, JP

REAL(KIND=JPRB) :: ZDP(KLON), ZPAPHD(KLON)

REAL(KIND=JPRB) :: ZALFA
! & ZALFACU, ZALFALS
REAL(KIND=JPRB) :: ZALFAW
REAL(KIND=JPRB) :: ZBETA,ZBETA1
!REAL(KIND=JPRB) :: ZBOTT
REAL(KIND=JPRB) :: ZCFPR
REAL(KIND=JPRB) :: ZCOR
REAL(KIND=JPRB) :: ZCDMAX
REAL(KIND=JPRB) :: ZMIN(KLON)
REAL(KIND=JPRB) :: ZLCONDLIM
REAL(KIND=JPRB) :: ZDENOM
REAL(KIND=JPRB) :: ZDPMXDT
REAL(KIND=JPRB) :: ZDPR
REAL(KIND=JPRB) :: ZDTDP
REAL(KIND=JPRB) :: ZE
REAL(KIND=JPRB) :: ZEPSEC
REAL(KIND=JPRB) :: ZFAC, ZFACI, ZFACW
REAL(KIND=JPRB) :: ZGDCP
REAL(KIND=JPRB) :: ZINEW
REAL(KIND=JPRB) :: ZLCRIT
REAL(KIND=JPRB) :: ZMFDN
REAL(KIND=JPRB) :: ZPRECIP
REAL(KIND=JPRB) :: ZQE
REAL(KIND=JPRB) :: ZQSAT, ZQTMST, ZRDCP
REAL(KIND=JPRB) :: ZRHC, ZSIG, ZSIGK
REAL(KIND=JPRB) :: ZWTOT
REAL(KIND=JPRB) :: ZZCO, ZZDL, ZZRH, ZZZDT, ZQADJ
REAL(KIND=JPRB) :: ZQNEW, ZTNEW
REAL(KIND=JPRB) :: ZRG_R,ZGDPH_R,ZCONS1,ZCOND,ZCONS1A
REAL(KIND=JPRB) :: ZLFINAL
REAL(KIND=JPRB) :: ZMELT
REAL(KIND=JPRB) :: ZEVAP
REAL(KIND=JPRB) :: ZFRZ
REAL(KIND=JPRB) :: ZVPLIQ, ZVPICE
REAL(KIND=JPRB) :: ZADD, ZBDD, ZCVDS, ZICE0, ZDEPOS
REAL(KIND=JPRB) :: ZSUPSAT(KLON)
REAL(KIND=JPRB) :: ZFALL
REAL(KIND=JPRB) :: ZRE_ICE
REAL(KIND=JPRB) :: ZRLDCP
REAL(KIND=JPRB) :: ZQP1ENV


!----------------------------
! Arrays for new microphysics
!----------------------------
INTEGER(KIND=JPIM) :: IPHASE(NCLV) ! marker for water phase of each species
                                   ! 0=vapour, 1=liquid, 2=ice

INTEGER(KIND=JPIM) :: IMELT(NCLV)  ! marks melting linkage for ice categories
                                   ! ice->liquid, snow->rain

LOGICAL :: LLFALL(NCLV)      ! marks falling species
                             ! LLFALL=0, cloud cover must > 0 for zqx > 0
                             ! LLFALL=1, no cloud needed, zqx can evaporate

LOGICAL            :: LLINDEX1(KLON,NCLV)      ! index variable
LOGICAL            :: LLINDEX3(KLON,NCLV,NCLV) ! index variable
REAL(KIND=JPRB)    :: ZMAX
REAL(KIND=JPRB)    :: ZRAT 
INTEGER(KIND=JPIM) :: IORDER(KLON,NCLV) ! array for sorting explicit terms

REAL(KIND=JPRB) :: ZLIQFRAC(KLON,KLEV)  ! cloud liquid water fraction: ql/(ql+qi)
REAL(KIND=JPRB) :: ZICEFRAC(KLON,KLEV)  ! cloud ice water fraction: qi/(ql+qi)
REAL(KIND=JPRB) :: ZQX(KLON,KLEV,NCLV)  ! water variables
REAL(KIND=JPRB) :: ZQX0(KLON,KLEV,NCLV) ! water variables at start of scheme
REAL(KIND=JPRB) :: ZQXN(KLON,NCLV)      ! new values for zqx at time+1
REAL(KIND=JPRB) :: ZQXFG(KLON,NCLV)     ! first guess values including precip
REAL(KIND=JPRB) :: ZQXNM1(KLON,NCLV)    ! new values for zqx at time+1 at level above
REAL(KIND=JPRB) :: ZFLUXQ(KLON,NCLV)    ! fluxes convergence of species (needed?)
! Keep the following for possible future total water variance scheme?
!REAL(KIND=JPRB) :: ZTL(KLON,KLEV)       ! liquid water temperature
!REAL(KIND=JPRB) :: ZABETA(KLON,KLEV)    ! cloud fraction
!REAL(KIND=JPRB) :: ZVAR(KLON,KLEV)      ! temporary variance
!REAL(KIND=JPRB) :: ZQTMIN(KLON,KLEV)
!REAL(KIND=JPRB) :: ZQTMAX(KLON,KLEV)

REAL(KIND=JPRB) :: ZPFPLSX(KLON,KLEV+1,NCLV) ! generalized precipitation flux
REAL(KIND=JPRB) :: ZLNEG(KLON,KLEV,NCLV)     ! for negative correction diagnostics
REAL(KIND=JPRB) :: ZMELTMAX(KLON)
REAL(KIND=JPRB) :: ZFRZMAX(KLON)
REAL(KIND=JPRB) :: ZICETOT(KLON)

REAL(KIND=JPRB) :: ZQXN2D(KLON,KLEV,NCLV)   ! water variables store

REAL(KIND=JPRB) :: ZQSMIX(KLON,KLEV) ! diagnostic mixed phase saturation 
!REAL(KIND=JPRB) :: ZQSBIN(KLON,KLEV) ! binary switched ice/liq saturation
REAL(KIND=JPRB) :: ZQSLIQ(KLON,KLEV) ! liquid water saturation
REAL(KIND=JPRB) :: ZQSICE(KLON,KLEV) ! ice water saturation

!REAL(KIND=JPRB) :: ZRHM(KLON,KLEV) ! diagnostic mixed phase RH
!REAL(KIND=JPRB) :: ZRHL(KLON,KLEV) ! RH wrt liq
!REAL(KIND=JPRB) :: ZRHI(KLON,KLEV) ! RH wrt ice

REAL(KIND=JPRB) :: ZFOEEWMT(KLON,KLEV)
REAL(KIND=JPRB) :: ZFOEEW(KLON,KLEV)
REAL(KIND=JPRB) :: ZFOEELIQT(KLON,KLEV)
!REAL(KIND=JPRB) :: ZFOEEICET(KLON,KLEV)

REAL(KIND=JPRB) :: ZDQSLIQDT(KLON), ZDQSICEDT(KLON), ZDQSMIXDT(KLON)
REAL(KIND=JPRB) :: ZCORQSLIQ(KLON)
REAL(KIND=JPRB) :: ZCORQSICE(KLON) 
!REAL(KIND=JPRB) :: ZCORQSBIN(KLON)
REAL(KIND=JPRB) :: ZCORQSMIX(KLON)
REAL(KIND=JPRB) :: ZEVAPLIMLIQ(KLON), ZEVAPLIMICE(KLON), ZEVAPLIMMIX(KLON)

!-------------------------------------------------------
! SOURCE/SINK array for implicit and explicit terms
!-------------------------------------------------------
! a POSITIVE value entered into the arrays is a...
!            Source of this variable
!            |
!            |   Sink of this variable
!            |   |
!            V   V
! ZSOLQA(JL,IQa,IQb)  = explicit terms
! ZSOLQB(JL,IQa,IQb)  = implicit terms
! Thus if ZSOLAB(JL,NCLDQL,IQV)=K where K>0 then this is 
! a source of NCLDQL and a sink of IQV
! put 'magic' source terms such as PLUDE from 
! detrainment into explicit source/sink array diagnognal
! ZSOLQA(NCLDQL,NCLDQL)= -PLUDE
! i.e. A positive value is a sink!????? weird... 
!-------------------------------------------------------

REAL(KIND=JPRB) :: ZSOLQA(KLON,NCLV,NCLV) ! explicit sources and sinks
REAL(KIND=JPRB) :: ZSOLQB(KLON,NCLV,NCLV) ! implicit sources and sinks
                        ! e.g. microphysical pathways between ice variables.
REAL(KIND=JPRB) :: ZQLHS(KLON,NCLV,NCLV)  ! n x n matrix storing the LHS of implicit solver
REAL(KIND=JPRB) :: ZVQX(NCLV)        ! fall speeds of three categories
REAL(KIND=JPRB) :: ZEXPLICIT, ZRATIO(KLON,NCLV), ZSINKSUM(KLON,NCLV)

! for sedimentation source/sink terms
REAL(KIND=JPRB) :: ZFALLSINK(KLON,NCLV)
REAL(KIND=JPRB) :: ZFALLSRCE(KLON,NCLV)

! for convection detrainment source and subsidence source/sink terms
REAL(KIND=JPRB) :: ZCONVSRCE(KLON,NCLV)
REAL(KIND=JPRB) :: ZCONVSINK(KLON,NCLV)

! for supersaturation source term from previous timestep
REAL(KIND=JPRB) :: ZPSUPSATSRCE(KLON,NCLV)

! Numerical fit to wet bulb temperature
REAL(KIND=JPRB),PARAMETER :: ZTW1 = 1329.31
REAL(KIND=JPRB),PARAMETER :: ZTW2 = 0.0074615
REAL(KIND=JPRB),PARAMETER :: ZTW3 = 0.85E5
REAL(KIND=JPRB),PARAMETER :: ZTW4 = 40.637
REAL(KIND=JPRB),PARAMETER :: ZTW5 = 275.0

REAL(KIND=JPRB) :: ZSUBSAT  ! Subsaturation for snow melting term         
REAL(KIND=JPRB) :: ZTDMTW0  ! Diff between dry-bulb temperature and 
                            ! temperature when wet-bulb = 0degC 

! Variables for deposition term
REAL(KIND=JPRB) :: ZTCG ! Temperature dependent function for ice PSD
REAL(KIND=JPRB) :: ZFACX1I, ZFACX1S! PSD correction factor
REAL(KIND=JPRB) :: ZAPLUSB,ZCORRFAC,ZCORRFAC2,ZPR02,ZTERM1,ZTERM2 ! for ice dep
REAL(KIND=JPRB) :: ZCLDTOPDIST(KLON) ! Distance from cloud top
REAL(KIND=JPRB) :: ZINFACTOR         ! No. of ice nuclei factor for deposition

! Autoconversion/accretion/riming/evaporation
INTEGER(KIND=JPIM) :: IWARMRAIN
INTEGER(KIND=JPIM) :: IEVAPRAIN
INTEGER(KIND=JPIM) :: IEVAPSNOW
INTEGER(KIND=JPIM) :: IDEPICE
REAL(KIND=JPRB) :: ZRAINACC(KLON)
REAL(KIND=JPRB) :: ZRAINCLD(KLON)
REAL(KIND=JPRB) :: ZSNOWRIME(KLON)
REAL(KIND=JPRB) :: ZSNOWCLD(KLON)
REAL(KIND=JPRB) :: ZESATLIQ
REAL(KIND=JPRB) :: ZFALLCORR
REAL(KIND=JPRB) :: ZLAMBDA
REAL(KIND=JPRB) :: ZEVAP_DENOM
REAL(KIND=JPRB) :: ZCORR2
REAL(KIND=JPRB) :: ZKA
REAL(KIND=JPRB) :: ZCONST
REAL(KIND=JPRB) :: ZTEMP

! Rain freezing
LOGICAL :: LLRAINLIQ(KLON)  ! True if majority of raindrops are liquid (no ice core)

!----------------------------
! End: new microphysics
!----------------------------
!REAL(KIND=JPRB), ALLOCATABLE :: ZSUMQ0(:,:),  ZSUMQ1(:,:) , ZERRORQ(:,:), &
!                               &ZSUMH0(:,:),  ZSUMH1(:,:) , ZERRORH(:,:)
!----------------------
! SCM budget statistics 
!----------------------
REAL(KIND=JPRB) :: ZRAIN

REAL(KIND=JPRB) :: Z_TMP1(KFDIA-KIDIA+1)
REAL(KIND=JPRB) :: Z_TMP2(KFDIA-KIDIA+1)
REAL(KIND=JPRB) :: Z_TMP3(KFDIA-KIDIA+1)
REAL(KIND=JPRB) :: Z_TMP4(KFDIA-KIDIA+1)
!REAL(KIND=JPRB) :: Z_TMP5(KFDIA-KIDIA+1)
REAL(KIND=JPRB) :: Z_TMP6(KFDIA-KIDIA+1)
REAL(KIND=JPRB) :: Z_TMP7(KFDIA-KIDIA+1)
REAL(KIND=JPRB) :: Z_TMPK(KFDIA-KIDIA+1,KLEV)
!REAL(KIND=JPRB) :: ZCON1,ZCON2
REAL(KIND=JPRB) :: ZHOOK_HANDLE
REAL(KIND=JPRB) :: ZTMPL,ZTMPI,ZTMPA

REAL(KIND=JPRB) :: ZMM,ZRR
REAL(KIND=JPRB) :: ZRG(KLON)

REAL(KIND=JPRB) :: ZBUDCC(KLON,KFLDX) ! extra fields
REAL(KIND=JPRB) :: ZBUDL(KLON,KFLDX) ! extra fields
REAL(KIND=JPRB) :: ZBUDI(KLON,KFLDX) ! extra fields

REAL(KIND=JPRB) :: ZZSUM, ZZRATIO
REAL(KIND=JPRB) :: ZEPSILON

REAL(KIND=JPRB) :: ZCOND1, ZQP
REAL(KIND=JPRB) :: PEXTRA(KLON,KLEV,KFLDX) ! extra fields


LOGICAL :: LLCLDBUDCC    ! Cloud fraction budget
LOGICAL :: LLCLDBUDL     ! Cloud liquid condensate budget
LOGICAL :: LLCLDBUDI     ! Cloud ice condensate budget



!===============================================================================
!IF (LHOOK) CALL DR_HOOK('CLOUDSC',0,ZHOOK_HANDLE)
!ASSOCIATE(LAERICEAUTO=>YRECLDP_LAERICEAUTO, LAERICESED=>YRECLDP_LAERICESED, &
! & LAERLIQAUTOLSP=>YRECLDP_LAERLIQAUTOLSP, LAERLIQCOLL=>YRECLDP_LAERLIQCOLL, &
! & LCLDBUDGET=>YRECLDP_LCLDBUDGET, NCLDTOP=>YRECLDP_NCLDTOP, &
! & NSSOPT=>YRECLDP_NSSOPT, RAMID=>YRECLDP_RAMID, RAMIN=>YRECLDP_RAMIN, &
! & RCCN=>YRECLDP_RCCN, RCLCRIT_LAND=>YRECLDP_RCLCRIT_LAND, &
! & RCLCRIT_SEA=>YRECLDP_RCLCRIT_SEA, RCLDIFF=>YRECLDP_RCLDIFF, &
! & RCLDIFF_CONVI=>YRECLDP_RCLDIFF_CONVI, RCLDTOPCF=>YRECLDP_RCLDTOPCF, &
! & RCL_APB1=>YRECLDP_RCL_APB1, RCL_APB2=>YRECLDP_RCL_APB2, &
! & RCL_APB3=>YRECLDP_RCL_APB3, RCL_CDENOM1=>YRECLDP_RCL_CDENOM1, &
! & RCL_CDENOM2=>YRECLDP_RCL_CDENOM2, RCL_CDENOM3=>YRECLDP_RCL_CDENOM3, &
! & RCL_CONST1I=>YRECLDP_RCL_CONST1I, RCL_CONST1R=>YRECLDP_RCL_CONST1R, &
! & RCL_CONST1S=>YRECLDP_RCL_CONST1S, RCL_CONST2I=>YRECLDP_RCL_CONST2I, &
! & RCL_CONST2R=>YRECLDP_RCL_CONST2R, RCL_CONST2S=>YRECLDP_RCL_CONST2S, &
! & RCL_CONST3I=>YRECLDP_RCL_CONST3I, RCL_CONST3R=>YRECLDP_RCL_CONST3R, &
! & RCL_CONST3S=>YRECLDP_RCL_CONST3S, RCL_CONST4I=>YRECLDP_RCL_CONST4I, &
! & RCL_CONST4R=>YRECLDP_RCL_CONST4R, RCL_CONST4S=>YRECLDP_RCL_CONST4S, &
! & RCL_CONST5I=>YRECLDP_RCL_CONST5I, RCL_CONST5R=>YRECLDP_RCL_CONST5R, &
! & RCL_CONST5S=>YRECLDP_RCL_CONST5S, RCL_CONST6I=>YRECLDP_RCL_CONST6I, &
! & RCL_CONST6R=>YRECLDP_RCL_CONST6R, RCL_CONST6S=>YRECLDP_RCL_CONST6S, &
! & RCL_CONST7S=>YRECLDP_RCL_CONST7S, RCL_CONST8S=>YRECLDP_RCL_CONST8S, &
! & RCL_FAC1=>YRECLDP_RCL_FAC1, RCL_FAC2=>YRECLDP_RCL_FAC2, &
! & RCL_FZRAB=>YRECLDP_RCL_FZRAB, RCL_KA273=>YRECLDP_RCL_KA273, &
! & RCL_KKAAC=>YRECLDP_RCL_KKAAC, RCL_KKAAU=>YRECLDP_RCL_KKAAU, &
! & RCL_KKBAC=>YRECLDP_RCL_KKBAC, RCL_KKBAUN=>YRECLDP_RCL_KKBAUN, &
! & RCL_KKBAUQ=>YRECLDP_RCL_KKBAUQ, &
! & RCL_KK_CLOUD_NUM_LAND=>YRECLDP_RCL_KK_CLOUD_NUM_LAND, &
! & RCL_KK_CLOUD_NUM_SEA=>YRECLDP_RCL_KK_CLOUD_NUM_SEA, RCL_X3I=>YRECLDP_RCL_X3I, &
! & RCOVPMIN=>YRECLDP_RCOVPMIN, RDENSREF=>YRECLDP_RDENSREF, &
! & RDEPLIQREFDEPTH=>YRECLDP_RDEPLIQREFDEPTH, &
! & RDEPLIQREFRATE=>YRECLDP_RDEPLIQREFRATE, RICEHI1=>YRECLDP_RICEHI1, &
! & RICEHI2=>YRECLDP_RICEHI2, RICEINIT=>YRECLDP_RICEINIT, RKCONV=>YRECLDP_RKCONV, &
! & RKOOPTAU=>YRECLDP_RKOOPTAU, RLCRITSNOW=>YRECLDP_RLCRITSNOW, &
! & RLMIN=>YRECLDP_RLMIN, RNICE=>YRECLDP_RNICE, RPECONS=>YRECLDP_RPECONS, &
! & RPRC1=>YRECLDP_RPRC1, RPRECRHMAX=>YRECLDP_RPRECRHMAX, &
! & RSNOWLIN1=>YRECLDP_RSNOWLIN1, RSNOWLIN2=>YRECLDP_RSNOWLIN2, &
! & RTAUMEL=>YRECLDP_RTAUMEL, RTHOMO=>YRECLDP_RTHOMO, RVICE=>YRECLDP_RVICE, &
! & RVRAIN=>YRECLDP_RVRAIN, RVRFACTOR=>YRECLDP_RVRFACTOR, &
! & RVSNOW=>YRECLDP_RVSNOW)

!===============================================================================
!  0.0     Beginning of timestep book-keeping
!----------------------------------------------------------------------


REAL(KIND=JPRB) :: FOEDELTA
REAL(KIND=JPRB) :: PTARE
REAL(KIND=JPRB) :: FOEEW,FOEDE,FOEDESU,FOELH,FOELDCP
REAL(KIND=JPRB) :: FOEALFA
REAL(KIND=JPRB) :: FOEEWM,FOEDEM,FOELDCPM,FOELHM,FOE_DEWM_DT
REAL(KIND=JPRB) :: FOETB
REAL(KIND=JPRB) :: FOEALFCU 
REAL(KIND=JPRB) :: FOEEWMCU,FOEDEMCU,FOELDCPMCU,FOELHMCU
REAL(KIND=JPRB) :: FOEEWMO, FOEELIQ, FOEEICE 
REAL(KIND=JPRB) :: FOEEWM_V,FOEEWMCU_V,FOELES_V,FOEIES_V
REAL(KIND=JPRB) :: EXP1,EXP2
REAL(KIND=JPRB) :: FOKOOP 

FOEDELTA (PTARE) = MAX (0.0,SIGN(1.0,PTARE-RTT))

!                  FOEDELTA = 1    water
!                  FOEDELTA = 0    ice

!     THERMODYNAMICAL FUNCTIONS .

!     Pressure of water vapour at saturation
!        INPUT : PTARE = TEMPERATURE

FOEEW ( PTARE ) = R2ES*EXP (&
  &(R3LES*FOEDELTA(PTARE)+R3IES*(1.0-FOEDELTA(PTARE)))*(PTARE-RTT)&
&/ (PTARE-(R4LES*FOEDELTA(PTARE)+R4IES*(1.0-FOEDELTA(PTARE)))))

FOEDE ( PTARE ) = &
  &(FOEDELTA(PTARE)*R5ALVCP+(1.0-FOEDELTA(PTARE))*R5ALSCP)&
&/ (PTARE-(R4LES*FOEDELTA(PTARE)+R4IES*(1.0-FOEDELTA(PTARE))))**2

FOEDESU ( PTARE ) = &
  &(FOEDELTA(PTARE)*R5LES+(1.0-FOEDELTA(PTARE))*R5IES)&
&/ (PTARE-(R4LES*FOEDELTA(PTARE)+R4IES*(1.0-FOEDELTA(PTARE))))**2

FOELH ( PTARE ) =&
         &FOEDELTA(PTARE)*RLVTT + (1.0-FOEDELTA(PTARE))*RLSTT

FOELDCP ( PTARE ) = &
         &FOEDELTA(PTARE)*RALVDCP + (1.0-FOEDELTA(PTARE))*RALSDCP

!     *****************************************************************

!           CONSIDERATION OF MIXED PHASES

!     *****************************************************************

!     FOEALFA is calculated to distinguish the three cases:

!                       FOEALFA=1            water phase
!                       FOEALFA=0            ice phase
!                       0 < FOEALFA < 1      mixed phase

!               INPUT : PTARE = TEMPERATURE


FOEALFA (PTARE) = MIN(1.0,((MAX(RTICE,MIN(RTWAT,PTARE))-RTICE)&
 &*RTWAT_RTICE_R)**2) 


!     Pressure of water vapour at saturation
!        INPUT : PTARE = TEMPERATURE

FOEEWM ( PTARE ) = R2ES *&
     &(FOEALFA(PTARE)*EXP(R3LES*(PTARE-RTT)/(PTARE-R4LES))+&
  &(1.0-FOEALFA(PTARE))*EXP(R3IES*(PTARE-RTT)/(PTARE-R4IES)))

FOE_DEWM_DT( PTARE ) = R2ES * ( &
     & R3LES*FOEALFA(PTARE)*EXP(R3LES*(PTARE-RTT)/(PTARE-R4LES)) &
     &    *(RTT-R4LES)/(PTARE-R4LES)**2 + &
     & R3IES*(1.0-FOEALFA(PTARE))*EXP(R3IES*(PTARE-RTT)/(PTARE-R4IES)) &
     &    *(RTT-R4IES)/(PTARE-R4IES)**2)

FOEDEM ( PTARE ) = FOEALFA(PTARE)*R5ALVCP*(1.0/(PTARE-R4LES)**2)+&
             &(1.0-FOEALFA(PTARE))*R5ALSCP*(1.0/(PTARE-R4IES)**2)

FOELDCPM ( PTARE ) = FOEALFA(PTARE)*RALVDCP+&
            &(1.0-FOEALFA(PTARE))*RALSDCP

FOELHM ( PTARE ) =&
         &FOEALFA(PTARE)*RLVTT+(1.0-FOEALFA(PTARE))*RLSTT


!     Temperature normalization for humidity background change of variable
!        INPUT : PTARE = TEMPERATURE

FOETB ( PTARE )=FOEALFA(PTARE)*R3LES*(RTT-R4LES)*(1.0/(PTARE-R4LES)**2)+&
             &(1.0-FOEALFA(PTARE))*R3IES*(RTT-R4IES)*(1.0/(PTARE-R4IES)**2)

!     ------------------------------------------------------------------
!     *****************************************************************

!           CONSIDERATION OF DIFFERENT MIXED PHASE FOR CONV

!     *****************************************************************

!     FOEALFCU is calculated to distinguish the three cases:

!                       FOEALFCU=1            water phase
!                       FOEALFCU=0            ice phase
!                       0 < FOEALFCU < 1      mixed phase

!               INPUT : PTARE = TEMPERATURE

FOEALFCU (PTARE) = MIN(1.0,((MAX(RTICECU,MIN(RTWAT,PTARE))&
&-RTICECU)*RTWAT_RTICECU_R)**2) 


!     Pressure of water vapour at saturation
!        INPUT : PTARE = TEMPERATURE

FOEEWMCU ( PTARE ) = R2ES *&
     &(FOEALFCU(PTARE)*EXP(R3LES*(PTARE-RTT)/(PTARE-R4LES))+&
  &(1.0-FOEALFCU(PTARE))*EXP(R3IES*(PTARE-RTT)/(PTARE-R4IES)))

FOEDEMCU ( PTARE )=FOEALFCU(PTARE)*R5ALVCP*(1.0/(PTARE-R4LES)**2)+&
             &(1.0-FOEALFCU(PTARE))*R5ALSCP*(1.0/(PTARE-R4IES)**2)

FOELDCPMCU ( PTARE ) = FOEALFCU(PTARE)*RALVDCP+&
            &(1.0-FOEALFCU(PTARE))*RALSDCP

FOELHMCU ( PTARE ) =&
         &FOEALFCU(PTARE)*RLVTT+(1.0-FOEALFCU(PTARE))*RLSTT
!     ------------------------------------------------------------------

!     Pressure of water vapour at saturation
!     This one is for the WMO definition of saturation, i.e. always
!     with respect to water.
!     
!     Duplicate to FOEELIQ and FOEEICE for separate ice variable
!     FOEELIQ always respect to water 
!     FOEEICE always respect to ice 
!     (could use FOEEW and FOEEWMO, but naming convention unclear)

FOEEWMO( PTARE ) = R2ES*EXP(R3LES*(PTARE-RTT)/(PTARE-R4LES))
FOEELIQ( PTARE ) = R2ES*EXP(R3LES*(PTARE-RTT)/(PTARE-R4LES))
FOEEICE( PTARE ) = R2ES*EXP(R3IES*(PTARE-RTT)/(PTARE-R4IES))
FOELES_V(PTARE)=R3LES*(PTARE-RTT)/(PTARE-R4LES)
FOEIES_V(PTARE)=R3IES*(PTARE-RTT)/(PTARE-R4IES)
FOEEWM_V( PTARE,EXP1,EXP2 )=R2ES*(FOEALFA(PTARE)*EXP1+ &
          & (1.0-FOEALFA(PTARE))*EXP2)
FOEEWMCU_V ( PTARE,EXP1,EXP2 ) = R2ES*(FOEALFCU(PTARE)*EXP1+&
          &(1.0-FOEALFCU(PTARE))*EXP2)
FOKOOP (PTARE) = MIN(RKOOP1-RKOOP2*PTARE,FOEELIQ(PTARE)/FOEEICE(PTARE))



!######################################################################
!             0.  *** SET UP CONSTANTS ***
!######################################################################

ZEPSILON=100.*EPSILON(ZEPSILON)
!ZEPSILON=0.0012919
! ---------------------------------------------------------------------
! Set version of warm-rain autoconversion/accretion
! IWARMRAIN = 1 ! Sundquist
! IWARMRAIN = 2 ! Khairoutdinov and Kogan (2000)
! ---------------------------------------------------------------------
IWARMRAIN = 2
! ---------------------------------------------------------------------
! Set version of rain evaporation
! IEVAPRAIN = 1 ! Sundquist
! IEVAPRAIN = 2 ! Abel and Boutle (2013)
! ---------------------------------------------------------------------
IEVAPRAIN = 2
! ---------------------------------------------------------------------
! Set version of snow evaporation
! IEVAPSNOW = 1 ! Sundquist
! IEVAPSNOW = 2 ! New
! ---------------------------------------------------------------------
IEVAPSNOW = 1
! ---------------------------------------------------------------------
! Set version of ice deposition
! IDEPICE = 1 ! Rotstayn (2001)
! IDEPICE = 2 ! New
! ---------------------------------------------------------------------
IDEPICE = 1

! ---------------------
! Some simple constants
! ---------------------
ZQTMST  = 1.0/PTSPHY
ZGDCP   = RG/RCPD
ZRDCP   = RD/RCPD
ZCONS1A = RCPD/(RLMLT*RG*RTAUMEL)
ZEPSEC  = 1.E-14
ZRG_R   = 1.0/RG
ZRLDCP  = 1.0/(RALSDCP-RALVDCP)

! Note: Defined in module/yoecldp.F90
! NCLDQL=1    ! liquid cloud water
! NCLDQI=2    ! ice cloud water
! NCLDQR=3    ! rain water
! NCLDQS=4    ! snow
! NCLDQV=5    ! vapour

! -----------------------------------------------
! Define species phase, 0=vapour, 1=liquid, 2=ice
! -----------------------------------------------
IPHASE(NCLDQV)=0
IPHASE(NCLDQL)=1
IPHASE(NCLDQR)=1
IPHASE(NCLDQI)=2
IPHASE(NCLDQS)=2

! ---------------------------------------------------
! Set up melting/freezing index, 
! if an ice category melts/freezes, where does it go?
! ---------------------------------------------------
IMELT(NCLDQV)=-99
IMELT(NCLDQL)=NCLDQI
IMELT(NCLDQR)=NCLDQS
IMELT(NCLDQI)=NCLDQR
IMELT(NCLDQS)=NCLDQR

! -----------------------------------------------
! INITIALIZATION OF OUTPUT TENDENCIES
! -----------------------------------------------
DO JK=1,KLEV
  DO JL=KIDIA,KFDIA
    tendency_loc_T(JL,JK)=0.0
    tendency_loc_q(JL,JK)=0.0
    tendency_loc_a(JL,JK)=0.0
  ENDDO
ENDDO
DO JM=1,NCLV-1
  DO JK=1,KLEV
    DO JL=KIDIA,KFDIA
      tendency_loc_cld(JL,JK,JM)=0.0
    ENDDO
  ENDDO
ENDDO

! -------------------------
! set up fall speeds in m/s
! -------------------------
ZVQX(NCLDQV)=0.0 
ZVQX(NCLDQL)=0.0 
ZVQX(NCLDQI)=RVICE 
ZVQX(NCLDQR)=RVRAIN
ZVQX(NCLDQS)=RVSNOW

LLFALL(:)=.FALSE.
DO JM=1,NCLV
  IF (ZVQX(JM)>0.0) LLFALL(JM)=.TRUE. ! falling species  
ENDDO
! Set LLFALL to false for ice (but ice still sediments!)
! Need to rationalise this at some point
LLFALL(NCLDQI)=.FALSE.

!######################################################################
!             1.  *** INITIAL VALUES FOR VARIABLES ***
!######################################################################


! ----------------------
! non CLV initialization 
! ----------------------
DO JK=1,KLEV
  DO JL=KIDIA,KFDIA
    ZTP1(JL,JK)        = PT(JL,JK)+PTSPHY*tendency_tmp_T(JL,JK)
    ZQX(JL,JK,NCLDQV)  = PQ(JL,JK)+PTSPHY*tendency_tmp_q(JL,JK) 
    ZQX0(JL,JK,NCLDQV) = PQ(JL,JK)+PTSPHY*tendency_tmp_q(JL,JK)
    ZA(JL,JK)          = PA(JL,JK)+PTSPHY*tendency_tmp_a(JL,JK)
    ZAORIG(JL,JK)      = PA(JL,JK)+PTSPHY*tendency_tmp_a(JL,JK)
  ENDDO
ENDDO

! -------------------------------------
! initialization for CLV family
! -------------------------------------
DO JM=1,NCLV-1
  DO JK=1,KLEV
    DO JL=KIDIA,KFDIA
      ZQX(JL,JK,JM)  = PCLV(JL,JK,JM)+PTSPHY*tendency_tmp_cld(JL,JK,JM)
      ZQX0(JL,JK,JM) = PCLV(JL,JK,JM)+PTSPHY*tendency_tmp_cld(JL,JK,JM)
    ENDDO
  ENDDO
ENDDO

!-------------
! zero arrays
!-------------
ZPFPLSX(:,:,:) = 0.0 ! precip fluxes
ZQXN2D(:,:,:)  = 0.0 ! end of timestep values in 2D
ZLNEG(:,:,:)   = 0.0 ! negative input check
PRAINFRAC_TOPRFZ(:) =0.0 ! rain fraction at top of refreezing layer
LLRAINLIQ(:) = .TRUE.  ! Assume all raindrops are liquid initially

! ----------------------------------------------------
! Tidy up very small cloud cover or total cloud water
! ----------------------------------------------------
DO JK=1,KLEV
  DO JL=KIDIA,KFDIA
    IF (ZQX(JL,JK,NCLDQL)+ZQX(JL,JK,NCLDQI)<RLMIN.OR.ZA(JL,JK)<RAMIN) THEN

      ! Evaporate small cloud liquid water amounts
      ZLNEG(JL,JK,NCLDQL) = ZLNEG(JL,JK,NCLDQL)+ZQX(JL,JK,NCLDQL)
      ZQADJ               = ZQX(JL,JK,NCLDQL)*ZQTMST
      tendency_loc_q(JL,JK)        = tendency_loc_q(JL,JK)+ZQADJ
      tendency_loc_T(JL,JK)        = tendency_loc_T(JL,JK)-RALVDCP*ZQADJ
      ZQX(JL,JK,NCLDQV)   = ZQX(JL,JK,NCLDQV)+ZQX(JL,JK,NCLDQL)
      ZQX(JL,JK,NCLDQL)   = 0.0

      ! Evaporate small cloud ice water amounts
      ZLNEG(JL,JK,NCLDQI) = ZLNEG(JL,JK,NCLDQI)+ZQX(JL,JK,NCLDQI)
      ZQADJ               = ZQX(JL,JK,NCLDQI)*ZQTMST
      tendency_loc_q(JL,JK)        = tendency_loc_q(JL,JK)+ZQADJ
      tendency_loc_T(JL,JK)        = tendency_loc_T(JL,JK)-RALSDCP*ZQADJ
      ZQX(JL,JK,NCLDQV)   = ZQX(JL,JK,NCLDQV)+ZQX(JL,JK,NCLDQI)
      ZQX(JL,JK,NCLDQI)   = 0.0

      ! Set cloud cover to zero
      ZA(JL,JK)           = 0.0

    ENDIF
  ENDDO
ENDDO

! ---------------------------------
! Tidy up small CLV variables
! ---------------------------------
!DIR$ IVDEP
DO JM=1,NCLV-1
!DIR$ IVDEP
  DO JK=1,KLEV
!DIR$ IVDEP
    DO JL=KIDIA,KFDIA
      IF (ZQX(JL,JK,JM)<RLMIN) THEN
        ZLNEG(JL,JK,JM) = ZLNEG(JL,JK,JM)+ZQX(JL,JK,JM)
        ZQADJ               = ZQX(JL,JK,JM)*ZQTMST
        tendency_loc_q(JL,JK)        = tendency_loc_q(JL,JK)+ZQADJ
        IF (IPHASE(JM)==1) tendency_loc_T(JL,JK) = tendency_loc_T(JL,JK)-RALVDCP*ZQADJ
        IF (IPHASE(JM)==2) tendency_loc_T(JL,JK) = tendency_loc_T(JL,JK)-RALSDCP*ZQADJ
        ZQX(JL,JK,NCLDQV)   = ZQX(JL,JK,NCLDQV)+ZQX(JL,JK,JM)
        ZQX(JL,JK,JM)       = 0.0
      ENDIF
    ENDDO
  ENDDO
ENDDO



! ------------------------------
! Define saturation values
! ------------------------------
DO JK=1,KLEV
  DO JL=KIDIA,KFDIA
    !----------------------------------------
    ! old *diagnostic* mixed phase saturation
    !---------------------------------------- 
    ZFOEALFA(JL,JK)=FOEALFA(ZTP1(JL,JK))
    ZFOEEWMT(JL,JK)=MIN(FOEEWM(ZTP1(JL,JK))/PAP(JL,JK),0.5)
    ZQSMIX(JL,JK)=ZFOEEWMT(JL,JK)
    ZQSMIX(JL,JK)=ZQSMIX(JL,JK)/(1.0-RETV*ZQSMIX(JL,JK))

    !---------------------------------------------
    ! ice saturation T<273K
    ! liquid water saturation for T>273K 
    !---------------------------------------------
    ZALFA=FOEDELTA(ZTP1(JL,JK))
    ZFOEEW(JL,JK)=MIN((ZALFA*FOEELIQ(ZTP1(JL,JK))+ &
         &  (1.0-ZALFA)*FOEEICE(ZTP1(JL,JK)))/PAP(JL,JK),0.5)
    ZFOEEW(JL,JK)=MIN(0.5,ZFOEEW(JL,JK))
    ZQSICE(JL,JK)=ZFOEEW(JL,JK)/(1.0-RETV*ZFOEEW(JL,JK))

    !----------------------------------
    ! liquid water saturation
    !---------------------------------- 
    ZFOEELIQT(JL,JK)=MIN(FOEELIQ(ZTP1(JL,JK))/PAP(JL,JK),0.5)
    ZQSLIQ(JL,JK)=ZFOEELIQT(JL,JK)
    ZQSLIQ(JL,JK)=ZQSLIQ(JL,JK)/(1.0-RETV*ZQSLIQ(JL,JK))

!   !----------------------------------
!   ! ice water saturation
!   !---------------------------------- 
!   ZFOEEICET(JL,JK)=MIN(FOEEICE(ZTP1(JL,JK))/PAP(JL,JK),0.5)
!   ZQSICE(JL,JK)=ZFOEEICET(JL,JK)
!   ZQSICE(JL,JK)=ZQSICE(JL,JK)/(1.0-RETV*ZQSICE(JL,JK))
  ENDDO

ENDDO

DO JK=1,KLEV
  DO JL=KIDIA,KFDIA


    !------------------------------------------
    ! Ensure cloud fraction is between 0 and 1
    !------------------------------------------
    ZA(JL,JK)=MAX(0.0,MIN(1.0,ZA(JL,JK)))

    !-------------------------------------------------------------------
    ! Calculate liq/ice fractions (no longer a diagnostic relationship)
    !-------------------------------------------------------------------
    ZLI(JL,JK)=ZQX(JL,JK,NCLDQL)+ZQX(JL,JK,NCLDQI)
    IF (ZLI(JL,JK)>RLMIN) THEN
      ZLIQFRAC(JL,JK)=ZQX(JL,JK,NCLDQL)/ZLI(JL,JK)
      ZICEFRAC(JL,JK)=1.0-ZLIQFRAC(JL,JK)
    ELSE
      ZLIQFRAC(JL,JK)=0.0
      ZICEFRAC(JL,JK)=0.0
    ENDIF

  ENDDO
ENDDO

!######################################################################
!        2.       *** CONSTANTS AND PARAMETERS ***
!######################################################################
!  Calculate L in updrafts of bl-clouds
!  Specify QS, P/PS for tropopause (for c2)
!  And initialize variables
!------------------------------------------


!-----------------------------
! Reset single level variables
!-----------------------------

ZANEWM1(:)  = 0.0
ZDA(:)      = 0.0
ZCOVPCLR(:) = 0.0
ZCOVPMAX(:) = 0.0  
ZCOVPTOT(:) = 0.0
ZCLDTOPDIST(:) = 0.0

!######################################################################
!           3.       *** PHYSICS ***
!######################################################################


!----------------------------------------------------------------------
!                       START OF VERTICAL LOOP
!----------------------------------------------------------------------

DO JK=NCLDTOP,KLEV

!----------------------------------------------------------------------
! 3.0 INITIALIZE VARIABLES
!----------------------------------------------------------------------

  !---------------------------------
  ! First guess microphysics
  !---------------------------------
  DO JM=1,NCLV
    DO JL=KIDIA,KFDIA
      ZQXFG(JL,JM)=ZQX(JL,JK,JM)
    ENDDO
  ENDDO

  !---------------------------------
  ! Set KLON arrays to zero
  !---------------------------------

  ZLICLD(:)   = 0.0                                
  ZRAINAUT(:) = 0.0  ! currently needed for diags  
  ZRAINACC(:) = 0.0  ! currently needed for diags  
  ZSNOWAUT(:) = 0.0  ! needed                      
  ZLDEFR(:)   = 0.0                                
  ZACUST(:)   = 0.0  ! set later when needed       
  ZQPRETOT(:) = 0.0                                
  ZLFINALSUM(:)= 0.0                               

  ! Required for first guess call
  ZLCOND1(:) = 0.0
  ZLCOND2(:) = 0.0
  ZSUPSAT(:) = 0.0
  ZLEVAPL(:) = 0.0
  ZLEVAPI(:) = 0.0

  !-------------------------------------                
  ! solvers for cloud fraction                          
  !-------------------------------------                
  ZSOLAB(:) = 0.0
  ZSOLAC(:) = 0.0

  !------------------------------------------           
  ! reset matrix so missing pathways are set            
  !------------------------------------------           
  ZSOLQB(:,:,:) = 0.0
  ZSOLQA(:,:,:) = 0.0

  !----------------------------------                   
  ! reset new microphysics variables                    
  !----------------------------------                   
  ZFALLSRCE(:,:) = 0.0
  ZFALLSINK(:,:) = 0.0
  ZCONVSRCE(:,:) = 0.0
  ZCONVSINK(:,:) = 0.0
  ZPSUPSATSRCE(:,:) = 0.0
  ZRATIO(:,:)    = 0.0
  ZICETOT(:)     = 0.0                            
  
  DO JL=KIDIA,KFDIA

    !-------------------------
    ! derived variables needed
    !-------------------------

    ZDP(JL)     = PAPH(JL,JK+1)-PAPH(JL,JK)     ! dp
    ZGDP(JL)    = RG/ZDP(JL)                    ! g/dp
    ZRHO(JL)    = PAP(JL,JK)/(RD*ZTP1(JL,JK))   ! p/RT air density

    ZDTGDP(JL)  = PTSPHY*ZGDP(JL)               ! dt g/dp
    ZRDTGDP(JL) = ZDP(JL)*(1.0/(PTSPHY*RG))  ! 1/(dt g/dp)

    IF (JK>1) ZDTGDPF(JL) = PTSPHY*RG/(PAP(JL,JK)-PAP(JL,JK-1))

    !------------------------------------
    ! Calculate dqs/dT correction factor
    !------------------------------------
    ! Reminder: RETV=RV/RD-1
    
    ! liquid
    ZFACW         = R5LES/((ZTP1(JL,JK)-R4LES)**2)
    ZCOR          = 1.0/(1.0-RETV*ZFOEELIQT(JL,JK))
    ZDQSLIQDT(JL) = ZFACW*ZCOR*ZQSLIQ(JL,JK)
    ZCORQSLIQ(JL) = 1.0+RALVDCP*ZDQSLIQDT(JL)

    ! ice
    ZFACI         = R5IES/((ZTP1(JL,JK)-R4IES)**2)
    ZCOR          = 1.0/(1.0-RETV*ZFOEEW(JL,JK))
    ZDQSICEDT(JL) = ZFACI*ZCOR*ZQSICE(JL,JK)
    ZCORQSICE(JL) = 1.0+RALSDCP*ZDQSICEDT(JL)

    ! diagnostic mixed
    ZALFAW        = ZFOEALFA(JL,JK)
    ZALFAWM(JL)   = ZALFAW
    ZFAC          = ZALFAW*ZFACW+(1.0-ZALFAW)*ZFACI
    ZCOR          = 1.0/(1.0-RETV*ZFOEEWMT(JL,JK))
    ZDQSMIXDT(JL) = ZFAC*ZCOR*ZQSMIX(JL,JK)
    ZCORQSMIX(JL) = 1.0+FOELDCPM(ZTP1(JL,JK))*ZDQSMIXDT(JL)

    ! evaporation/sublimation limits
    ZEVAPLIMMIX(JL) = MAX((ZQSMIX(JL,JK)-ZQX(JL,JK,NCLDQV))/ZCORQSMIX(JL),0.0)
    ZEVAPLIMLIQ(JL) = MAX((ZQSLIQ(JL,JK)-ZQX(JL,JK,NCLDQV))/ZCORQSLIQ(JL),0.0)
    ZEVAPLIMICE(JL) = MAX((ZQSICE(JL,JK)-ZQX(JL,JK,NCLDQV))/ZCORQSICE(JL),0.0)

    !--------------------------------
    ! in-cloud consensate amount
    !--------------------------------
    ZTMPA = 1.0/MAX(ZA(JL,JK),ZEPSEC)
    ZLIQCLD(JL) = ZQX(JL,JK,NCLDQL)*ZTMPA
    ZICECLD(JL) = ZQX(JL,JK,NCLDQI)*ZTMPA
    ZLICLD(JL)  = ZLIQCLD(JL)+ZICECLD(JL)

  ENDDO
  
  !------------------------------------------------
  ! Evaporate very small amounts of liquid and ice
  !------------------------------------------------
  DO JL=KIDIA,KFDIA

    IF (ZQX(JL,JK,NCLDQL) < RLMIN) THEN
      ZSOLQA(JL,NCLDQV,NCLDQL) = ZQX(JL,JK,NCLDQL)
      ZSOLQA(JL,NCLDQL,NCLDQV) = -ZQX(JL,JK,NCLDQL)
    ENDIF

    IF (ZQX(JL,JK,NCLDQI) < RLMIN) THEN
      ZSOLQA(JL,NCLDQV,NCLDQI) = ZQX(JL,JK,NCLDQI)
      ZSOLQA(JL,NCLDQI,NCLDQV) = -ZQX(JL,JK,NCLDQI)
    ENDIF

  ENDDO
  
  !---------------------------------------------------------------------
  !  3.1  ICE SUPERSATURATION ADJUSTMENT
  !---------------------------------------------------------------------
  ! Note that the supersaturation adjustment is made with respect to 
  ! liquid saturation:  when T>0C 
  ! ice saturation:     when T<0C
  !                     with an adjustment made to allow for ice 
  !                     supersaturation in the clear sky
  ! Note also that the KOOP factor automatically clips the supersaturation
  ! to a maximum set by the liquid water saturation mixing ratio
  ! important for temperatures near to but below 0C
  !----------------------------------------------------------------------- 

!DIR$ NOFUSION
  DO JL=KIDIA,KFDIA

    !-----------------------------------
    ! 3.1.1 Supersaturation limit (from Koop)
    !-----------------------------------
    ! Needs to be set for all temperatures
    ZFOKOOP(JL)=FOKOOP(ZTP1(JL,JK))
  ENDDO
  DO JL=KIDIA,KFDIA

    IF (ZTP1(JL,JK)>=RTT .OR. NSSOPT==0) THEN
      ZFAC  = 1.0
      ZFACI = 1.0
    ELSE
      ZFAC  = ZA(JL,JK)+ZFOKOOP(JL)*(1.0-ZA(JL,JK))
      ZFACI = PTSPHY/RKOOPTAU
    ENDIF

    !-------------------------------------------------------------------
    ! 3.1.2 Calculate supersaturation wrt Koop including dqs/dT 
    !       correction factor
    ! [#Note: QSICE or QSLIQ]
    !-------------------------------------------------------------------

    ! Calculate supersaturation to add to cloud
    IF (ZA(JL,JK) > 1.0-RAMIN) THEN
      ZSUPSAT(JL) = MAX((ZQX(JL,JK,NCLDQV)-ZFAC*ZQSICE(JL,JK))/ZCORQSICE(JL)&
     &                  ,0.0)
    ELSE
      ! Calculate environmental humidity supersaturation
      ZQP1ENV = (ZQX(JL,JK,NCLDQV) - ZA(JL,JK)*ZQSICE(JL,JK))/ &
     & MAX(1.0-ZA(JL,JK),ZEPSILON)
    !& SIGN(MAX(ABS(1.0-ZA(JL,JK)),ZEPSILON),1.0-ZA(JL,JK))
      ZSUPSAT(JL) = MAX((1.0-ZA(JL,JK))*(ZQP1ENV-ZFAC*ZQSICE(JL,JK))&
     &                  /ZCORQSICE(JL),0.0)
    ENDIF 
    
    !-------------------------------------------------------------------
    ! Here the supersaturation is turned into liquid water
    ! However, if the temperature is below the threshold for homogeneous
    ! freezing then the supersaturation is turned instantly to ice.
    !--------------------------------------------------------------------

    IF (ZSUPSAT(JL) > ZEPSEC) THEN

      IF (ZTP1(JL,JK) > RTHOMO) THEN
        ! Turn supersaturation into liquid water        
        ZSOLQA(JL,NCLDQL,NCLDQV) = ZSOLQA(JL,NCLDQL,NCLDQV)+ZSUPSAT(JL)
        ZSOLQA(JL,NCLDQV,NCLDQL) = ZSOLQA(JL,NCLDQV,NCLDQL)-ZSUPSAT(JL)
        ! Include liquid in first guess
        ZQXFG(JL,NCLDQL)=ZQXFG(JL,NCLDQL)+ZSUPSAT(JL)
      ELSE
        ! Turn supersaturation into ice water        
        ZSOLQA(JL,NCLDQI,NCLDQV) = ZSOLQA(JL,NCLDQI,NCLDQV)+ZSUPSAT(JL)
        ZSOLQA(JL,NCLDQV,NCLDQI) = ZSOLQA(JL,NCLDQV,NCLDQI)-ZSUPSAT(JL)
        ! Add ice to first guess for deposition term 
        ZQXFG(JL,NCLDQI)=ZQXFG(JL,NCLDQI)+ZSUPSAT(JL)
      ENDIF

      ! Increase cloud amount using RKOOPTAU timescale
      ZSOLAC(JL) = (1.0-ZA(JL,JK))*ZFACI

   ENDIF

    !-------------------------------------------------------
    ! 3.1.3 Include supersaturation from previous timestep
    ! (Calculated in sltENDIF semi-lagrangian LDSLPHY=T)
    !-------------------------------------------------------    
      IF (PSUPSAT(JL,JK)>ZEPSEC) THEN
        IF (ZTP1(JL,JK) > RTHOMO) THEN
          ! Turn supersaturation into liquid water
          ZSOLQA(JL,NCLDQL,NCLDQL) = ZSOLQA(JL,NCLDQL,NCLDQL)+PSUPSAT(JL,JK)
          ZPSUPSATSRCE(JL,NCLDQL) = PSUPSAT(JL,JK)
          ! Add liquid to first guess for deposition term 
          ZQXFG(JL,NCLDQL)=ZQXFG(JL,NCLDQL)+PSUPSAT(JL,JK)
          ! Store cloud budget diagnostics if required
        ELSE
          ! Turn supersaturation into ice water
          ZSOLQA(JL,NCLDQI,NCLDQI) = ZSOLQA(JL,NCLDQI,NCLDQI)+PSUPSAT(JL,JK)
          ZPSUPSATSRCE(JL,NCLDQI) = PSUPSAT(JL,JK)
          ! Add ice to first guess for deposition term 
          ZQXFG(JL,NCLDQI)=ZQXFG(JL,NCLDQI)+PSUPSAT(JL,JK)
          ! Store cloud budget diagnostics if required
        ENDIF

        ! Increase cloud amount using RKOOPTAU timescale
        ZSOLAC(JL)=(1.0-ZA(JL,JK))*ZFACI
        ! Store cloud budget diagnostics if required
      ENDIF

  ENDDO ! on JL

  !---------------------------------------------------------------------
  !  3.2  DETRAINMENT FROM CONVECTION
  !---------------------------------------------------------------------
  ! * Diagnostic T-ice/liq split retained for convection
  !    Note: This link is now flexible and a future convection 
  !    scheme can detrain explicit seperate budgets of:
  !    cloud water, ice, rain and snow
  ! * There is no (1-ZA) multiplier term on the cloud detrainment 
  !    term, since is now written in mass-flux terms  
  ! [#Note: Should use ZFOEALFACU used in convection rather than ZFOEALFA]
  !---------------------------------------------------------------------
  IF (JK < KLEV .AND. JK>=NCLDTOP) THEN

    DO JL=KIDIA,KFDIA
    
      PLUDE(JL,JK)=PLUDE(JL,JK)*ZDTGDP(JL)

      IF(LDCUM(JL).AND.PLUDE(JL,JK) > RLMIN.AND.PLU(JL,JK+1)> ZEPSEC) THEN
    
        ZSOLAC(JL)=ZSOLAC(JL)+PLUDE(JL,JK)/PLU(JL,JK+1)
        ! *diagnostic temperature split*
        ZALFAW               = ZFOEALFA(JL,JK)
        ZCONVSRCE(JL,NCLDQL) = ZALFAW*PLUDE(JL,JK)
        ZCONVSRCE(JL,NCLDQI) = (1.0-ZALFAW)*PLUDE(JL,JK)
        ZSOLQA(JL,NCLDQL,NCLDQL) = ZSOLQA(JL,NCLDQL,NCLDQL)+ZCONVSRCE(JL,NCLDQL)
        ZSOLQA(JL,NCLDQI,NCLDQI) = ZSOLQA(JL,NCLDQI,NCLDQI)+ZCONVSRCE(JL,NCLDQI)
        
      ELSE

        PLUDE(JL,JK)=0.0
    
      ENDIF
        ! *convective snow detrainment source
      IF (LDCUM(JL)) ZSOLQA(JL,NCLDQS,NCLDQS) = ZSOLQA(JL,NCLDQS,NCLDQS) + PSNDE(JL,JK)*ZDTGDP(JL)
    
    ENDDO

  ENDIF ! JK<KLEV

  !---------------------------------------------------------------------
  !  3.3  SUBSIDENCE COMPENSATING CONVECTIVE UPDRAUGHTS
  !---------------------------------------------------------------------
  ! Three terms:
  ! * Convective subsidence source of cloud from layer above
  ! * Evaporation of cloud within the layer
  ! * Subsidence sink of cloud to the layer below (Implicit solution)
  !---------------------------------------------------------------------

  !-----------------------------------------------
  ! Subsidence source from layer above
  !               and 
  ! Evaporation of cloud within the layer
  !-----------------------------------------------
  IF (JK > NCLDTOP) THEN

    DO JL=KIDIA,KFDIA
      ZMF(JL)=MAX(0.0,(PMFU(JL,JK)+PMFD(JL,JK))*ZDTGDP(JL))
      ZACUST(JL)=ZMF(JL)*ZANEWM1(JL)
    ENDDO

    DO JM=1,NCLV
      IF (.NOT.LLFALL(JM).AND.IPHASE(JM)>0) THEN 
        DO JL=KIDIA,KFDIA
          ZLCUST(JL,JM)=ZMF(JL)*ZQXNM1(JL,JM)
          ! record total flux for enthalpy budget:
          ZCONVSRCE(JL,JM)=ZCONVSRCE(JL,JM)+ZLCUST(JL,JM)
        ENDDO
      ENDIF
    ENDDO

    ! Now have to work out how much liquid evaporates at arrival point 
    ! since there is no prognostic memory for in-cloud humidity, i.e. 
    ! we always assume cloud is saturated. 

    DO JL=KIDIA,KFDIA
      ZDTDP=ZRDCP*0.5*(ZTP1(JL,JK-1)+ZTP1(JL,JK))/PAPH(JL,JK)
      ZDTFORC = ZDTDP*(PAP(JL,JK)-PAP(JL,JK-1))
      ![#Note: Diagnostic mixed phase should be replaced below]
      ZDQS(JL)=ZANEWM1(JL)*ZDTFORC*ZDQSMIXDT(JL)
    ENDDO

    DO JM=1,NCLV
      IF (.NOT.LLFALL(JM).AND.IPHASE(JM)>0) THEN 
        DO JL=KIDIA,KFDIA
          ZLFINAL=MAX(0.0,ZLCUST(JL,JM)-ZDQS(JL)) !lim to zero
          ! no supersaturation allowed incloud ---V
          ZEVAP=MIN((ZLCUST(JL,JM)-ZLFINAL),ZEVAPLIMMIX(JL)) 
!          ZEVAP=0.0
          ZLFINAL=ZLCUST(JL,JM)-ZEVAP 
          ZLFINALSUM(JL)=ZLFINALSUM(JL)+ZLFINAL ! sum 

          ZSOLQA(JL,JM,JM)     = ZSOLQA(JL,JM,JM)+ZLCUST(JL,JM) ! whole sum 
          ZSOLQA(JL,NCLDQV,JM) = ZSOLQA(JL,NCLDQV,JM)+ZEVAP
          ZSOLQA(JL,JM,NCLDQV) = ZSOLQA(JL,JM,NCLDQV)-ZEVAP
        ENDDO
      ENDIF
    ENDDO

    !  Reset the cloud contribution if no cloud water survives to this level:
    DO JL=KIDIA,KFDIA
      IF (ZLFINALSUM(JL)<ZEPSEC) ZACUST(JL)=0.0
      ZSOLAC(JL)=ZSOLAC(JL)+ZACUST(JL)
    ENDDO

  ENDIF ! on  JK>NCLDTOP

  !---------------------------------------------------------------------
  ! Subsidence sink of cloud to the layer below 
  ! (Implicit - re. CFL limit on convective mass flux)
  !---------------------------------------------------------------------

  DO JL=KIDIA,KFDIA

    IF(JK<KLEV) THEN

      ZMFDN=MAX(0.0,(PMFU(JL,JK+1)+PMFD(JL,JK+1))*ZDTGDP(JL))
      
      ZSOLAB(JL)=ZSOLAB(JL)+ZMFDN
      ZSOLQB(JL,NCLDQL,NCLDQL)=ZSOLQB(JL,NCLDQL,NCLDQL)+ZMFDN
      ZSOLQB(JL,NCLDQI,NCLDQI)=ZSOLQB(JL,NCLDQI,NCLDQI)+ZMFDN

      ! Record sink for cloud budget and enthalpy budget diagnostics
      ZCONVSINK(JL,NCLDQL) = ZMFDN
      ZCONVSINK(JL,NCLDQI) = ZMFDN

    ENDIF

  ENDDO

  !----------------------------------------------------------------------
  ! 3.4  EROSION OF CLOUDS BY TURBULENT MIXING
  !----------------------------------------------------------------------
  ! NOTE: In default tiedtke scheme this process decreases the cloud 
  !       area but leaves the specific cloud water content
  !       within clouds unchanged
  !----------------------------------------------------------------------

  ! ------------------------------
  ! Define turbulent erosion rate
  ! ------------------------------
  DO JL=KIDIA,KFDIA
    ZLDIFDT(JL)=RCLDIFF*PTSPHY !original version
    !Increase by factor of 5 for convective points
    IF(KTYPE(JL) > 0 .AND. PLUDE(JL,JK) > ZEPSEC)&
       & ZLDIFDT(JL)=RCLDIFF_CONVI*ZLDIFDT(JL)  
  ENDDO

  ! At the moment, works on mixed RH profile and partitioned ice/liq fraction
  ! so that it is similar to previous scheme
  ! Should apply RHw for liquid cloud and RHi for ice cloud separately 
  DO JL=KIDIA,KFDIA
    IF(ZLI(JL,JK) > ZEPSEC) THEN
      ! Calculate environmental humidity
!      ZQE=(ZQX(JL,JK,NCLDQV)-ZA(JL,JK)*ZQSMIX(JL,JK))/&
!    &      MAX(ZEPSEC,1.0-ZA(JL,JK))  
!      ZE=ZLDIFDT(JL)*MAX(ZQSMIX(JL,JK)-ZQE,0.0)
      ZE=ZLDIFDT(JL)*MAX(ZQSMIX(JL,JK)-ZQX(JL,JK,NCLDQV),0.0)
      ZLEROS=ZA(JL,JK)*ZE
      ZLEROS=MIN(ZLEROS,ZEVAPLIMMIX(JL))
      ZLEROS=MIN(ZLEROS,ZLI(JL,JK))
      ZAEROS=ZLEROS/ZLICLD(JL)  !if linear term

      ! Erosion is -ve LINEAR in L,A
      ZSOLAC(JL)=ZSOLAC(JL)-ZAEROS !linear

      ZSOLQA(JL,NCLDQV,NCLDQL) = ZSOLQA(JL,NCLDQV,NCLDQL)+ZLIQFRAC(JL,JK)*ZLEROS
      ZSOLQA(JL,NCLDQL,NCLDQV) = ZSOLQA(JL,NCLDQL,NCLDQV)-ZLIQFRAC(JL,JK)*ZLEROS
      ZSOLQA(JL,NCLDQV,NCLDQI) = ZSOLQA(JL,NCLDQV,NCLDQI)+ZICEFRAC(JL,JK)*ZLEROS
      ZSOLQA(JL,NCLDQI,NCLDQV) = ZSOLQA(JL,NCLDQI,NCLDQV)-ZICEFRAC(JL,JK)*ZLEROS

    ENDIF
  ENDDO

  !----------------------------------------------------------------------
  ! 3.4  CONDENSATION/EVAPORATION DUE TO DQSAT/DT
  !----------------------------------------------------------------------
  !  calculate dqs/dt
  !  Note: For the separate prognostic Qi and Ql, one would ideally use
  !  Qsat/DT wrt liquid/Koop here, since the physics is that new clouds
  !  forms by liquid droplets [liq] or when aqueous aerosols [Koop] form.
  !  These would then instantaneous freeze if T<-38C or lead to ice growth 
  !  by deposition in warmer mixed phase clouds.  However, since we do 
  !  not have a separate prognostic equation for in-cloud humidity or a 
  !  statistical scheme approach in place, the depositional growth of ice 
  !  in the mixed phase can not be modelled and we resort to supersaturation  
  !  wrt ice instanteously converting to ice over one timestep 
  !  (see Tompkins et al. QJRMS 2007 for details)
  !  Thus for the initial implementation the diagnostic mixed phase is 
  !  retained for the moment, and the level of approximation noted.  
  !----------------------------------------------------------------------

  DO JL=KIDIA,KFDIA
    ZDTDP   = ZRDCP*ZTP1(JL,JK)/PAP(JL,JK)
    ZDPMXDT = ZDP(JL)*ZQTMST
    ZMFDN   = 0.0
    IF(JK < KLEV) ZMFDN=PMFU(JL,JK+1)+PMFD(JL,JK+1)
    ZWTOT   = PVERVEL(JL,JK)+0.5*RG*(PMFU(JL,JK)+PMFD(JL,JK)+ZMFDN)
    ZWTOT   = MIN(ZDPMXDT,MAX(-ZDPMXDT,ZWTOT))
    ZZZDT   = PHRSW(JL,JK)+PHRLW(JL,JK)
    ZDTDIAB = MIN(ZDPMXDT*ZDTDP,MAX(-ZDPMXDT*ZDTDP,ZZZDT))&
                    & *PTSPHY+RALFDCP*ZLDEFR(JL)  
! Note: ZLDEFR should be set to the difference between the mixed phase functions
! in the convection and cloud scheme, but this is not calculated, so is zero and
! the functions must be the same
    ZDTFORC = ZDTDP*ZWTOT*PTSPHY+ZDTDIAB
    ZQOLD(JL)   = ZQSMIX(JL,JK)
    ZTOLD(JL)   = ZTP1(JL,JK)
    ZTP1(JL,JK) = ZTP1(JL,JK)+ZDTFORC
    ZTP1(JL,JK) = MAX(ZTP1(JL,JK),160.0)
    LLFLAG(JL)  = .TRUE.
  ENDDO

  ! Formerly a call to CUADJTQ(..., ICALL=5)
  DO JL=KIDIA,KFDIA
     ZQP   = 1.0/PAP(JL,JK)
     ZQSAT = FOEEWM(ZTP1(JL,JK))*ZQP
     ZQSAT = MIN(0.5,ZQSAT)
     ZCOR  = 1.0/(1.0-RETV  *ZQSAT)
     ZQSAT = ZQSAT*ZCOR
     ZCOND = (ZQSMIX(JL,JK)-ZQSAT)/(1.0+ZQSAT*ZCOR*FOEDEM(ZTP1(JL,JK)))
     ZTP1(JL,JK) = ZTP1(JL,JK)+FOELDCPM(ZTP1(JL,JK))*ZCOND
     ZQSMIX(JL,JK) = ZQSMIX(JL,JK)-ZCOND
     ZQSAT = FOEEWM(ZTP1(JL,JK))*ZQP
     ZQSAT = MIN(0.5,ZQSAT)
     ZCOR  = 1.0/(1.0-RETV  *ZQSAT)
     ZQSAT = ZQSAT*ZCOR
     ZCOND1= (ZQSMIX(JL,JK)-ZQSAT)/(1.0+ZQSAT*ZCOR*FOEDEM(ZTP1(JL,JK)))
     ZTP1(JL,JK) = ZTP1(JL,JK)+FOELDCPM(ZTP1(JL,JK))*ZCOND1
     ZQSMIX(JL,JK) = ZQSMIX(JL,JK)-ZCOND1
  ENDDO

  DO JL=KIDIA,KFDIA
    ZDQS(JL)      = ZQSMIX(JL,JK)-ZQOLD(JL)
    ZQSMIX(JL,JK) = ZQOLD(JL)
    ZTP1(JL,JK)   = ZTOLD(JL)
  ENDDO

  !----------------------------------------------------------------------
  ! 3.4a  ZDQS(JL) > 0:  EVAPORATION OF CLOUDS
  ! ----------------------------------------------------------------------
  ! Erosion term is LINEAR in L
  ! Changed to be uniform distribution in cloud region

  DO JL=KIDIA,KFDIA

    ! Previous function based on DELTA DISTRIBUTION in cloud:
   IF (ZDQS(JL) > 0.0) THEN
!    If subsidence evaporation term is turned off, then need to use updated
!    liquid and cloud here?
!    ZLEVAP = MAX(ZA(JL,JK)+ZACUST(JL),1.0)*MIN(ZDQS(JL),ZLICLD(JL)+ZLFINALSUM(JL))
    ZLEVAP = ZA(JL,JK)*MIN(ZDQS(JL),ZLICLD(JL))
    ZLEVAP = MIN(ZLEVAP,ZEVAPLIMMIX(JL))
    ZLEVAP = MIN(ZLEVAP,MAX(ZQSMIX(JL,JK)-ZQX(JL,JK,NCLDQV),0.0))

    ! For first guess call
    ZLEVAPL(JL) = ZLIQFRAC(JL,JK)*ZLEVAP
    ZLEVAPI(JL) = ZICEFRAC(JL,JK)*ZLEVAP

    ZSOLQA(JL,NCLDQV,NCLDQL) = ZSOLQA(JL,NCLDQV,NCLDQL)+ZLIQFRAC(JL,JK)*ZLEVAP
    ZSOLQA(JL,NCLDQL,NCLDQV) = ZSOLQA(JL,NCLDQL,NCLDQV)-ZLIQFRAC(JL,JK)*ZLEVAP

    ZSOLQA(JL,NCLDQV,NCLDQI) = ZSOLQA(JL,NCLDQV,NCLDQI)+ZICEFRAC(JL,JK)*ZLEVAP
    ZSOLQA(JL,NCLDQI,NCLDQV) = ZSOLQA(JL,NCLDQI,NCLDQV)-ZICEFRAC(JL,JK)*ZLEVAP

   ENDIF

  ENDDO

  !----------------------------------------------------------------------
  ! 3.4b ZDQS(JL) < 0: FORMATION OF CLOUDS
  !----------------------------------------------------------------------
  ! (1) Increase of cloud water in existing clouds
  DO JL=KIDIA,KFDIA
    IF(ZA(JL,JK) > ZEPSEC.AND.ZDQS(JL) <= -RLMIN) THEN

      ZLCOND1(JL)=MAX(-ZDQS(JL),0.0) !new limiter

!old limiter (significantly improves upper tropospheric humidity rms)
      IF(ZA(JL,JK) > 0.99) THEN
        ZCOR=1.0/(1.0-RETV*ZQSMIX(JL,JK))
        ZCDMAX=(ZQX(JL,JK,NCLDQV)-ZQSMIX(JL,JK))/&
         & (1.0+ZCOR*ZQSMIX(JL,JK)*FOEDEM(ZTP1(JL,JK)))  
      ELSE
        ZCDMAX=(ZQX(JL,JK,NCLDQV)-ZA(JL,JK)*ZQSMIX(JL,JK))/ZA(JL,JK)
      ENDIF
      ZLCOND1(JL)=MAX(MIN(ZLCOND1(JL),ZCDMAX),0.0)
! end old limiter
      
      ZLCOND1(JL)=ZA(JL,JK)*ZLCOND1(JL)
      IF(ZLCOND1(JL) < RLMIN) ZLCOND1(JL)=0.0
      
      !-------------------------------------------------------------------------
      ! All increase goes into liquid unless so cold cloud homogeneously freezes
      ! Include new liquid formation in first guess value, otherwise liquid 
      ! remains at cold temperatures until next timestep.
      !-------------------------------------------------------------------------
      IF (ZTP1(JL,JK)>RTHOMO) THEN
        ZSOLQA(JL,NCLDQL,NCLDQV)=ZSOLQA(JL,NCLDQL,NCLDQV)+ZLCOND1(JL)
        ZSOLQA(JL,NCLDQV,NCLDQL)=ZSOLQA(JL,NCLDQV,NCLDQL)-ZLCOND1(JL)
        ZQXFG(JL,NCLDQL)=ZQXFG(JL,NCLDQL)+ZLCOND1(JL)
      ELSE
        ZSOLQA(JL,NCLDQI,NCLDQV)=ZSOLQA(JL,NCLDQI,NCLDQV)+ZLCOND1(JL)
        ZSOLQA(JL,NCLDQV,NCLDQI)=ZSOLQA(JL,NCLDQV,NCLDQI)-ZLCOND1(JL)
        ZQXFG(JL,NCLDQI)=ZQXFG(JL,NCLDQI)+ZLCOND1(JL)
      ENDIF
    ENDIF
  ENDDO

  ! (2) Generation of new clouds (da/dt>0)
  
  DO JL=KIDIA,KFDIA

    IF(ZDQS(JL) <= -RLMIN .AND. ZA(JL,JK)<1.0-ZEPSEC) THEN

      !---------------------------
      ! Critical relative humidity
      !---------------------------
      ZRHC=RAMID
      ZSIGK=PAP(JL,JK)/PAPH(JL,KLEV+1)
      ! Increase RHcrit to 1.0 towards the surface (eta>0.8)
      IF(ZSIGK > 0.8) THEN
        ZRHC=RAMID+(1.0-RAMID)*((ZSIGK-0.8)/0.2)**2
      ENDIF

! Commented out for CY37R1 to reduce humidity in high trop and strat
!      ! Increase RHcrit to 1.0 towards the tropopause (trop-0.2) and above
!      ZBOTT=ZTRPAUS(JL)+0.2
!      IF(ZSIGK < ZBOTT) THEN
!        ZRHC=RAMID+(1.0-RAMID)*MIN(((ZBOTT-ZSIGK)/0.2)**2,1.0)
!      ENDIF

      !---------------------------
      ! Supersaturation options
      !---------------------------      
      IF (NSSOPT==0) THEN 
        ! No scheme
        ZQE=(ZQX(JL,JK,NCLDQV)-ZA(JL,JK)*ZQSICE(JL,JK))/&
            & MAX(ZEPSEC,1.0-ZA(JL,JK))  
        ZQE=MAX(0.0,ZQE)
      ELSEIF (NSSOPT==1) THEN 
        ! Tompkins 
        ZQE=(ZQX(JL,JK,NCLDQV)-ZA(JL,JK)*ZQSICE(JL,JK))/&
            & MAX(ZEPSEC,1.0-ZA(JL,JK))  
        ZQE=MAX(0.0,ZQE)
      ELSEIF (NSSOPT==2) THEN 
        ! Lohmann and Karcher
        ZQE=ZQX(JL,JK,NCLDQV)  
      ELSEIF (NSSOPT==3) THEN 
        ! Gierens
        ZQE=ZQX(JL,JK,NCLDQV)+ZLI(JL,JK)
      ENDIF

      IF (ZTP1(JL,JK)>=RTT .OR. NSSOPT==0) THEN 
        ! No ice supersaturation allowed
        ZFAC=1.0        
      ELSE
        ! Ice supersaturation
        ZFAC=ZFOKOOP(JL)
      ENDIF

      IF(ZQE >= ZRHC*ZQSICE(JL,JK)*ZFAC.AND.ZQE<ZQSICE(JL,JK)*ZFAC) THEN
        ! note: not **2 on 1-a term if ZQE is used. 
        ! Added correction term ZFAC to numerator 15/03/2010
        ZACOND=-(1.0-ZA(JL,JK))*ZFAC*ZDQS(JL)/&
         &MAX(2.0*(ZFAC*ZQSICE(JL,JK)-ZQE),ZEPSEC)

        ZACOND=MIN(ZACOND,1.0-ZA(JL,JK))  !PUT THE LIMITER BACK

        ! Linear term:
        ! Added correction term ZFAC 15/03/2010
        ZLCOND2(JL)=-ZFAC*ZDQS(JL)*0.5*ZACOND !mine linear

        ! new limiter formulation
        ZZDL=2.0*(ZFAC*ZQSICE(JL,JK)-ZQE)/MAX(ZEPSEC,1.0-ZA(JL,JK))
        ! Added correction term ZFAC 15/03/2010
        IF (ZFAC*ZDQS(JL)<-ZZDL) THEN
          ! ZLCONDLIM=(ZA(JL,JK)-1.0)*ZDQS(JL)-ZQSICE(JL,JK)+ZQX(JL,JK,NCLDQV)
          ZLCONDLIM=(ZA(JL,JK)-1.0)*ZFAC*ZDQS(JL)- &
     &               ZFAC*ZQSICE(JL,JK)+ZQX(JL,JK,NCLDQV)
          ZLCOND2(JL)=MIN(ZLCOND2(JL),ZLCONDLIM)
        ENDIF
        ZLCOND2(JL)=MAX(ZLCOND2(JL),0.0)

        IF(ZLCOND2(JL) < RLMIN .OR. (1.0-ZA(JL,JK))<ZEPSEC ) THEN
          ZLCOND2(JL) = 0.0
          ZACOND      = 0.0
        ENDIF
        IF(ZLCOND2(JL) == 0.0) ZACOND=0.0

        ! Large-scale generation is LINEAR in A and LINEAR in L
        ZSOLAC(JL) = ZSOLAC(JL)+ZACOND !linear
        
        !------------------------------------------------------------------------
        ! All increase goes into liquid unless so cold cloud homogeneously freezes
        ! Include new liquid formation in first guess value, otherwise liquid 
        ! remains at cold temperatures until next timestep.
        !------------------------------------------------------------------------
        IF (ZTP1(JL,JK)>RTHOMO) THEN
          ZSOLQA(JL,NCLDQL,NCLDQV)=ZSOLQA(JL,NCLDQL,NCLDQV)+ZLCOND2(JL)
          ZSOLQA(JL,NCLDQV,NCLDQL)=ZSOLQA(JL,NCLDQV,NCLDQL)-ZLCOND2(JL)
          ZQXFG(JL,NCLDQL)=ZQXFG(JL,NCLDQL)+ZLCOND2(JL)
        ELSE ! homogeneous freezing
          ZSOLQA(JL,NCLDQI,NCLDQV)=ZSOLQA(JL,NCLDQI,NCLDQV)+ZLCOND2(JL)
          ZSOLQA(JL,NCLDQV,NCLDQI)=ZSOLQA(JL,NCLDQV,NCLDQI)-ZLCOND2(JL)
          ZQXFG(JL,NCLDQI)=ZQXFG(JL,NCLDQI)+ZLCOND2(JL)
        ENDIF

      ENDIF
    ENDIF
  ENDDO

  !----------------------------------------------------------------------
  ! 3.7 Growth of ice by vapour deposition 
  !----------------------------------------------------------------------
  ! Following Rotstayn et al. 2001:
  ! does not use the ice nuclei number from cloudaer.F90
  ! but rather a simple Meyers et al. 1992 form based on the 
  ! supersaturation and assuming clouds are saturated with 
  ! respect to liquid water (well mixed), (or Koop adjustment)
  ! Growth considered as sink of liquid water if present so 
  ! Bergeron-Findeisen adjustment in autoconversion term no longer needed
  !----------------------------------------------------------------------

  !--------------------------------------------------------
  !-
  !- Ice deposition following Rotstayn et al. (2001)
  !-  (monodisperse ice particle size distribution)
  !-
  !--------------------------------------------------------
  IF (IDEPICE == 1) THEN
  
  DO JL=KIDIA,KFDIA

    !--------------------------------------------------------------
    ! Calculate distance from cloud top 
    ! defined by cloudy layer below a layer with cloud frac <0.01
    ! ZDZ = ZDP(JL)/(ZRHO(JL)*RG)
    !--------------------------------------------------------------
      
    IF (ZA(JL,JK-1) < RCLDTOPCF .AND. ZA(JL,JK) >= RCLDTOPCF) THEN
      ZCLDTOPDIST(JL) = 0.0
    ELSE
      ZCLDTOPDIST(JL) = ZCLDTOPDIST(JL) + ZDP(JL)/(ZRHO(JL)*RG)
    ENDIF

    !--------------------------------------------------------------
    ! only treat depositional growth if liquid present. due to fact 
    ! that can not model ice growth from vapour without additional 
    ! in-cloud water vapour variable
    !--------------------------------------------------------------
    IF (ZTP1(JL,JK)<RTT .AND. ZQXFG(JL,NCLDQL)>RLMIN) THEN  ! T<273K

      ZVPICE=FOEEICE(ZTP1(JL,JK))*RV/RD
      ZVPLIQ=ZVPICE*ZFOKOOP(JL) 
      ZICENUCLEI(JL)=1000.0*EXP(12.96*(ZVPLIQ-ZVPICE)/ZVPLIQ-0.639)

      !------------------------------------------------
      !   2.4e-2 is conductivity of air
      !   8.8 = 700**1/3 = density of ice to the third
      !------------------------------------------------
      ZADD=RLSTT*(RLSTT/(RV*ZTP1(JL,JK))-1.0)/(2.4E-2*ZTP1(JL,JK))
      ZBDD=RV*ZTP1(JL,JK)*PAP(JL,JK)/(2.21*ZVPICE)
      ZCVDS=7.8*(ZICENUCLEI(JL)/ZRHO(JL))**0.666*(ZVPLIQ-ZVPICE) / &
         & (8.87*(ZADD+ZBDD)*ZVPICE)

      !-----------------------------------------------------
      ! RICEINIT=1.E-12 is initial mass of ice particle
      !-----------------------------------------------------
      ZICE0=MAX(ZICECLD(JL), ZICENUCLEI(JL)*RICEINIT/ZRHO(JL))

      !------------------
      ! new value of ice:
      !------------------
      ZINEW=(0.666*ZCVDS*PTSPHY+ZICE0**0.666)**1.5

      !---------------------------
      ! grid-mean deposition rate:
      !--------------------------- 
      ZDEPOS=MAX(ZA(JL,JK)*(ZINEW-ZICE0),0.0)

      !--------------------------------------------------------------------
      ! Limit deposition to liquid water amount
      ! If liquid is all frozen, ice would use up reservoir of water 
      ! vapour in excess of ice saturation mixing ratio - However this 
      ! can not be represented without a in-cloud humidity variable. Using 
      ! the grid-mean humidity would imply a large artificial horizontal 
      ! flux from the clear sky to the cloudy area. We thus rely on the 
      ! supersaturation check to clean up any remaining supersaturation
      !--------------------------------------------------------------------
      ZDEPOS=MIN(ZDEPOS,ZQXFG(JL,NCLDQL)) ! limit to liquid water amount
      
      !--------------------------------------------------------------------
      ! At top of cloud, reduce deposition rate near cloud top to account for
      ! small scale turbulent processes, limited ice nucleation and ice fallout 
      !--------------------------------------------------------------------
!      ZDEPOS = ZDEPOS*MIN(RDEPLIQREFRATE+ZCLDTOPDIST(JL)/RDEPLIQREFDEPTH,1.0)
      ! Change to include dependence on ice nuclei concentration
      ! to increase deposition rate with decreasing temperatures 
      ZINFACTOR = MIN(ZICENUCLEI(JL)/15000., 1.0)
      ZDEPOS = ZDEPOS*MIN(ZINFACTOR + (1.0-ZINFACTOR)* &
                  & (RDEPLIQREFRATE+ZCLDTOPDIST(JL)/RDEPLIQREFDEPTH),1.0)

      !--------------
      ! add to matrix 
      !--------------
      ZSOLQA(JL,NCLDQI,NCLDQL)=ZSOLQA(JL,NCLDQI,NCLDQL)+ZDEPOS
      ZSOLQA(JL,NCLDQL,NCLDQI)=ZSOLQA(JL,NCLDQL,NCLDQI)-ZDEPOS
      ZQXFG(JL,NCLDQI)=ZQXFG(JL,NCLDQI)+ZDEPOS
      ZQXFG(JL,NCLDQL)=ZQXFG(JL,NCLDQL)-ZDEPOS

    ENDIF
  ENDDO
  
  !--------------------------------------------------------
  !-
  !- Ice deposition assuming ice PSD
  !-
  !--------------------------------------------------------
  ELSEIF (IDEPICE == 2) THEN

    DO JL=KIDIA,KFDIA

      !--------------------------------------------------------------
      ! Calculate distance from cloud top 
      ! defined by cloudy layer below a layer with cloud frac <0.01
      ! ZDZ = ZDP(JL)/(ZRHO(JL)*RG)
      !--------------------------------------------------------------

      IF (ZA(JL,JK-1) < RCLDTOPCF .AND. ZA(JL,JK) >= RCLDTOPCF) THEN
        ZCLDTOPDIST(JL) = 0.0
      ELSE
        ZCLDTOPDIST(JL) = ZCLDTOPDIST(JL) + ZDP(JL)/(ZRHO(JL)*RG)
      ENDIF

      !--------------------------------------------------------------
      ! only treat depositional growth if liquid present. due to fact 
      ! that can not model ice growth from vapour without additional 
      ! in-cloud water vapour variable
      !--------------------------------------------------------------
      IF (ZTP1(JL,JK)<RTT .AND. ZQXFG(JL,NCLDQL)>RLMIN) THEN  ! T<273K
      
        ZVPICE = FOEEICE(ZTP1(JL,JK))*RV/RD
        ZVPLIQ = ZVPICE*ZFOKOOP(JL) 
        ZICENUCLEI(JL)=1000.0*EXP(12.96*(ZVPLIQ-ZVPICE)/ZVPLIQ-0.639)

        !-----------------------------------------------------
        ! RICEINIT=1.E-12 is initial mass of ice particle
        !-----------------------------------------------------
        ZICE0=MAX(ZICECLD(JL), ZICENUCLEI(JL)*RICEINIT/ZRHO(JL))
        
        ! Particle size distribution
        ZTCG    = 1.0
        ZFACX1I = 1.0

        ZAPLUSB   = RCL_APB1*ZVPICE-RCL_APB2*ZVPICE*ZTP1(JL,JK)+ &
       &             PAP(JL,JK)*RCL_APB3*ZTP1(JL,JK)**3.
        ZCORRFAC  = (1.0/ZRHO(JL))**0.5
        ZCORRFAC2 = ((ZTP1(JL,JK)/273.0)**1.5) &
       &             *(393.0/(ZTP1(JL,JK)+120.0))

        ZPR02  = ZRHO(JL)*ZICE0*RCL_CONST1I/(ZTCG*ZFACX1I)

        ZTERM1 = (ZVPLIQ-ZVPICE)*ZTP1(JL,JK)**2.0*ZVPICE*ZCORRFAC2*ZTCG* &
       &          RCL_CONST2I*ZFACX1I/(ZRHO(JL)*ZAPLUSB*ZVPICE)
        ZTERM2 = 0.65*RCL_CONST6I*ZPR02**RCL_CONST4I+RCL_CONST3I &
       &          *ZCORRFAC**0.5*ZRHO(JL)**0.5 &
       &          *ZPR02**RCL_CONST5I/ZCORRFAC2**0.5

        ZDEPOS = MAX(ZA(JL,JK)*ZTERM1*ZTERM2*PTSPHY,0.0)

        !--------------------------------------------------------------------
        ! Limit deposition to liquid water amount
        ! If liquid is all frozen, ice would use up reservoir of water 
        ! vapour in excess of ice saturation mixing ratio - However this 
        ! can not be represented without a in-cloud humidity variable. Using 
        ! the grid-mean humidity would imply a large artificial horizontal 
        ! flux from the clear sky to the cloudy area. We thus rely on the 
        ! supersaturation check to clean up any remaining supersaturation
        !--------------------------------------------------------------------
        ZDEPOS=MIN(ZDEPOS,ZQXFG(JL,NCLDQL)) ! limit to liquid water amount

        !--------------------------------------------------------------------
        ! At top of cloud, reduce deposition rate near cloud top to account for
        ! small scale turbulent processes, limited ice nucleation and ice fallout 
        !--------------------------------------------------------------------
        ! Change to include dependence on ice nuclei concentration
        ! to increase deposition rate with decreasing temperatures 
        ZINFACTOR = MIN(ZICENUCLEI(JL)/15000., 1.0)
        ZDEPOS = ZDEPOS*MIN(ZINFACTOR + (1.0-ZINFACTOR)* &
                    & (RDEPLIQREFRATE+ZCLDTOPDIST(JL)/RDEPLIQREFDEPTH),1.0)

        !--------------
        ! add to matrix 
        !--------------
        ZSOLQA(JL,NCLDQI,NCLDQL) = ZSOLQA(JL,NCLDQI,NCLDQL)+ZDEPOS
        ZSOLQA(JL,NCLDQL,NCLDQI) = ZSOLQA(JL,NCLDQL,NCLDQI)-ZDEPOS
        ZQXFG(JL,NCLDQI) = ZQXFG(JL,NCLDQI)+ZDEPOS
        ZQXFG(JL,NCLDQL) = ZQXFG(JL,NCLDQL)-ZDEPOS
      ENDIF
    ENDDO

  ENDIF ! on IDEPICE
 
  !######################################################################
  !              4  *** PRECIPITATION PROCESSES ***
  !######################################################################

  !----------------------------------
  ! revise in-cloud consensate amount
  !----------------------------------
  DO JL=KIDIA,KFDIA
    ZTMPA = 1.0/MAX(ZA(JL,JK),ZEPSEC)
    ZLIQCLD(JL) = ZQXFG(JL,NCLDQL)*ZTMPA
    ZICECLD(JL) = ZQXFG(JL,NCLDQI)*ZTMPA
    ZLICLD(JL)  = ZLIQCLD(JL)+ZICECLD(JL)
  ENDDO

  !----------------------------------------------------------------------
  ! 4.2 SEDIMENTATION/FALLING OF *ALL* MICROPHYSICAL SPECIES
  !     now that rain, snow, graupel species are prognostic
  !     the precipitation flux can be defined directly level by level
  !     There is no vertical memory required from the flux variable
  !----------------------------------------------------------------------

  DO JM = 1,NCLV
    IF (LLFALL(JM) .OR. JM == NCLDQI) THEN
      DO JL=KIDIA,KFDIA
        !------------------------
        ! source from layer above 
        !------------------------
        IF (JK > NCLDTOP) THEN
          ZFALLSRCE(JL,JM) = ZPFPLSX(JL,JK,JM)*ZDTGDP(JL) 
          ZSOLQA(JL,JM,JM) = ZSOLQA(JL,JM,JM)+ZFALLSRCE(JL,JM)
          ZQXFG(JL,JM)     = ZQXFG(JL,JM)+ZFALLSRCE(JL,JM)
          ! use first guess precip----------V
          ZQPRETOT(JL)     = ZQPRETOT(JL)+ZQXFG(JL,JM) 
        ENDIF
        !-------------------------------------------------
        ! sink to next layer, constant fall speed
        !-------------------------------------------------
        ! if aerosol effect then override 
        !  note that for T>233K this is the same as above.
        IF (LAERICESED .AND. JM == NCLDQI) THEN
          ZRE_ICE=PRE_ICE(JL,JK) 
          ! The exponent value is from 
          ! Morrison et al. JAS 2005 Appendix
          ZVQX(NCLDQI) = 0.002*ZRE_ICE**1.0
        ENDIF
        ZFALL=ZVQX(JM)*ZRHO(JL)
        !-------------------------------------------------
        ! modified by Heymsfield and Iaquinta JAS 2000
        !-------------------------------------------------
        ! ZFALL = ZFALL*((PAP(JL,JK)*RICEHI1)**(-0.178)) &
        !            &*((ZTP1(JL,JK)*RICEHI2)**(-0.394))

        ZFALLSINK(JL,JM)=ZDTGDP(JL)*ZFALL
        ! Cloud budget diagnostic stored at end as implicit
      ENDDO ! jl  
    ENDIF ! LLFALL
  ENDDO ! jm

  !---------------------------------------------------------------
  ! Precip cover overlap using MAX-RAN Overlap
  ! Since precipitation is now prognostic we must 
  !   1) apply an arbitrary minimum coverage (0.3) if precip>0
  !   2) abandon the 2-flux clr/cld treatment
  !   3) Thus, since we have no memory of the clear sky precip
  !      fraction, we mimic the previous method by reducing 
  !      ZCOVPTOT(JL), which has the memory, proportionally with 
  !      the precip evaporation rate, taking cloud fraction 
  !      into account
  !   #3 above leads to much smoother vertical profiles of 
  !   precipitation fraction than the Klein-Jakob scheme which 
  !   monotonically increases precip fraction and then resets 
  !   it to zero in a step function once clear-sky precip reaches
  !   zero.
  !---------------------------------------------------------------
  DO JL=KIDIA,KFDIA
    IF (ZQPRETOT(JL)>ZEPSEC) THEN
      ZCOVPTOT(JL) = 1.0 - ((1.0-ZCOVPTOT(JL))*&
       &            (1.0 - MAX(ZA(JL,JK),ZA(JL,JK-1)))/&
       &            (1.0 - MIN(ZA(JL,JK-1),1.0-1.E-06)) )  
      ZCOVPTOT(JL) = MAX(ZCOVPTOT(JL),RCOVPMIN)
      ZCOVPCLR(JL) = MAX(0.0,ZCOVPTOT(JL)-ZA(JL,JK)) ! clear sky proportion
      ZRAINCLD(JL) = ZQXFG(JL,NCLDQR)/ZCOVPTOT(JL)
      ZSNOWCLD(JL) = ZQXFG(JL,NCLDQS)/ZCOVPTOT(JL)
      ZCOVPMAX(JL) = MAX(ZCOVPTOT(JL),ZCOVPMAX(JL))
    ELSE
      ZRAINCLD(JL) = 0.0 
      ZSNOWCLD(JL) = 0.0 
      ZCOVPTOT(JL) = 0.0 ! no flux - reset cover
      ZCOVPCLR(JL) = 0.0   ! reset clear sky proportion 
      ZCOVPMAX(JL) = 0.0 ! reset max cover for ZZRH calc 
    ENDIF
  ENDDO
  
  !----------------------------------------------------------------------
  ! 4.3a AUTOCONVERSION TO SNOW
  !----------------------------------------------------------------------
  DO JL=KIDIA,KFDIA
 
    IF(ZTP1(JL,JK) <= RTT) THEN
      !-----------------------------------------------------
      !     Snow Autoconversion rate follow Lin et al. 1983
      !-----------------------------------------------------
      IF (ZICECLD(JL)>ZEPSEC) THEN

        ZZCO=PTSPHY*RSNOWLIN1*EXP(RSNOWLIN2*(ZTP1(JL,JK)-RTT))

        IF (LAERICEAUTO) THEN
          ZLCRIT=PICRIT_AER(JL,JK)
          ! 0.3 = N**0.333 with N=0.027 
          ZZCO=ZZCO*(RNICE/PNICE(JL,JK))**0.333
        ELSE
          ZLCRIT=RLCRITSNOW
        ENDIF

        ZSNOWAUT(JL)=ZZCO*(1.0-EXP(-(ZICECLD(JL)/ZLCRIT)**2))
        ZSOLQB(JL,NCLDQS,NCLDQI)=ZSOLQB(JL,NCLDQS,NCLDQI)+ZSNOWAUT(JL)

      ENDIF
    ENDIF 
  
  !----------------------------------------------------------------------
  ! 4.3b AUTOCONVERSION WARM CLOUDS
  !   Collection and accretion will require separate treatment
  !   but for now we keep this simple treatment
  !----------------------------------------------------------------------

   IF (ZLIQCLD(JL)>ZEPSEC) THEN

    !--------------------------------------------------------
    !-
    !- Warm-rain process follow Sundqvist (1989)
    !-
    !--------------------------------------------------------
    IF (IWARMRAIN == 1) THEN

      ZZCO=RKCONV*PTSPHY

      IF (LAERLIQAUTOLSP) THEN
        ZLCRIT=PLCRIT_AER(JL,JK)
        ! 0.3 = N**0.333 with N=125 cm-3 
        ZZCO=ZZCO*(RCCN/PCCN(JL,JK))**0.333
      ELSE
        ! Modify autoconversion threshold dependent on: 
        !  land (polluted, high CCN, smaller droplets, higher threshold)
        !  sea  (clean, low CCN, larger droplets, lower threshold)
        IF (PLSM(JL) > 0.5) THEN
          ZLCRIT = RCLCRIT_LAND ! land
        ELSE
          ZLCRIT = RCLCRIT_SEA  ! ocean
        ENDIF
      ENDIF 

      !------------------------------------------------------------------
      ! Parameters for cloud collection by rain and snow.
      ! Note that with new prognostic variable it is now possible 
      ! to REPLACE this with an explicit collection parametrization
      !------------------------------------------------------------------   
      ZPRECIP=(ZPFPLSX(JL,JK,NCLDQS)+ZPFPLSX(JL,JK,NCLDQR))/MAX(ZEPSEC,ZCOVPTOT(JL))
      ZCFPR=1.0 + RPRC1*SQRT(MAX(ZPRECIP,0.0))
!      ZCFPR=1.0 + RPRC1*SQRT(MAX(ZPRECIP,0.0))*&
!       &ZCOVPTOT(JL)/(MAX(ZA(JL,JK),ZEPSEC))

      IF (LAERLIQCOLL) THEN 
        ! 5.0 = N**0.333 with N=125 cm-3 
        ZCFPR=ZCFPR*(RCCN/PCCN(JL,JK))**0.333
      ENDIF

      ZZCO=ZZCO*ZCFPR
      ZLCRIT=ZLCRIT/MAX(ZCFPR,ZEPSEC)
  
      IF(ZLIQCLD(JL)/ZLCRIT < 20.0 )THEN ! Security for exp for some compilers
        ZRAINAUT(JL)=ZZCO*(1.0-EXP(-(ZLIQCLD(JL)/ZLCRIT)**2))
      ELSE
        ZRAINAUT(JL)=ZZCO
      ENDIF

      ! rain freezes instantly
      IF(ZTP1(JL,JK) <= RTT) THEN
        ZSOLQB(JL,NCLDQS,NCLDQL)=ZSOLQB(JL,NCLDQS,NCLDQL)+ZRAINAUT(JL)
      ELSE
        ZSOLQB(JL,NCLDQR,NCLDQL)=ZSOLQB(JL,NCLDQR,NCLDQL)+ZRAINAUT(JL)
      ENDIF

    !--------------------------------------------------------
    !-
    !- Warm-rain process follow Khairoutdinov and Kogan (2000)
    !-
    !--------------------------------------------------------
    ELSEIF (IWARMRAIN == 2) THEN

      IF (PLSM(JL) > 0.5) THEN ! land
        ZCONST = RCL_KK_CLOUD_NUM_LAND
        ZLCRIT = RCLCRIT_LAND
      ELSE                          ! ocean
        ZCONST = RCL_KK_CLOUD_NUM_SEA
        ZLCRIT = RCLCRIT_SEA
      ENDIF
 
      IF (ZLIQCLD(JL) > ZLCRIT) THEN

        ZRAINAUT(JL)  = 1.5*ZA(JL,JK)*PTSPHY* &
     &                  RCL_KKAau * ZLIQCLD(JL)**RCL_KKBauq * ZCONST**RCL_KKBaun

        ZRAINAUT(JL) = MIN(ZRAINAUT(JL),ZQXFG(JL,NCLDQL))
        IF (ZRAINAUT(JL) < ZEPSEC) ZRAINAUT(JL) = 0.0

        ZRAINACC(JL) = 2.0*ZA(JL,JK)*PTSPHY* &
     &                 RCL_KKAac * (ZLIQCLD(JL)*ZRAINCLD(JL))**RCL_KKBac

        ZRAINACC(JL) = MIN(ZRAINACC(JL),ZQXFG(JL,NCLDQL))
        IF (ZRAINACC(JL) < ZEPSEC) ZRAINACC(JL) = 0.0

      ELSE
        ZRAINAUT(JL)  = 0.0
        ZRAINACC(JL)  = 0.0
      ENDIF

      ! If temperature < 0, then autoconversion produces snow rather than rain
      ! Explicit
      IF(ZTP1(JL,JK) <= RTT) THEN
        ZSOLQA(JL,NCLDQS,NCLDQL)=ZSOLQA(JL,NCLDQS,NCLDQL)+ZRAINAUT(JL)
        ZSOLQA(JL,NCLDQS,NCLDQL)=ZSOLQA(JL,NCLDQS,NCLDQL)+ZRAINACC(JL)
        ZSOLQA(JL,NCLDQL,NCLDQS)=ZSOLQA(JL,NCLDQL,NCLDQS)-ZRAINAUT(JL)
        ZSOLQA(JL,NCLDQL,NCLDQS)=ZSOLQA(JL,NCLDQL,NCLDQS)-ZRAINACC(JL)
      ELSE
        ZSOLQA(JL,NCLDQR,NCLDQL)=ZSOLQA(JL,NCLDQR,NCLDQL)+ZRAINAUT(JL)
        ZSOLQA(JL,NCLDQR,NCLDQL)=ZSOLQA(JL,NCLDQR,NCLDQL)+ZRAINACC(JL)
        ZSOLQA(JL,NCLDQL,NCLDQR)=ZSOLQA(JL,NCLDQL,NCLDQR)-ZRAINAUT(JL)
        ZSOLQA(JL,NCLDQL,NCLDQR)=ZSOLQA(JL,NCLDQL,NCLDQR)-ZRAINACC(JL)
      ENDIF
    
    ENDIF ! on IWARMRAIN
    
   ENDIF ! on ZLIQCLD > ZEPSEC
  ENDDO


  !----------------------------------------------------------------------
  ! RIMING - COLLECTION OF CLOUD LIQUID DROPS BY SNOW AND ICE
  !      only active if T<0degC and supercooled liquid water is present
  !      AND if not Sundquist autoconversion (as this includes riming)
  !----------------------------------------------------------------------
  IF (IWARMRAIN > 1) THEN

  DO JL=KIDIA,KFDIA
    IF(ZTP1(JL,JK) <= RTT .AND. ZLIQCLD(JL)>ZEPSEC) THEN

      ! Fallspeed air density correction 
      ZFALLCORR = (RDENSREF/ZRHO(JL))**0.4

      !------------------------------------------------------------------
      ! Riming of snow by cloud water - implicit in lwc
      !------------------------------------------------------------------
      IF (ZSNOWCLD(JL)>ZEPSEC .AND. ZCOVPTOT(JL)>0.01) THEN

        ! Calculate riming term
        ! Factor of liq water taken out because implicit
        ZSNOWRIME(JL) = 0.3*ZCOVPTOT(JL)*PTSPHY*RCL_CONST7S*ZFALLCORR &
     &                  *(ZRHO(JL)*ZSNOWCLD(JL)*RCL_CONST1S)**RCL_CONST8S

        ! Limit snow riming term
        ZSNOWRIME(JL)=MIN(ZSNOWRIME(JL),1.0)

        ZSOLQB(JL,NCLDQS,NCLDQL) = ZSOLQB(JL,NCLDQS,NCLDQL) + ZSNOWRIME(JL)

      ENDIF

      !------------------------------------------------------------------
      ! Riming of ice by cloud water - implicit in lwc
      ! NOT YET ACTIVE
      !------------------------------------------------------------------
!      IF (ZICECLD(JL)>ZEPSEC .AND. ZA(JL,JK)>0.01) THEN
!
!        ! Calculate riming term
!        ! Factor of liq water taken out because implicit
!        ZSNOWRIME(JL) = ZA(JL,JK)*PTSPHY*RCL_CONST7S*ZFALLCORR &
!     &                  *(ZRHO(JL)*ZICECLD(JL)*RCL_CONST1S)**RCL_CONST8S
!
!        ! Limit ice riming term
!        ZSNOWRIME(JL)=MIN(ZSNOWRIME(JL),1.0)
!
!        ZSOLQB(JL,NCLDQI,NCLDQL) = ZSOLQB(JL,NCLDQI,NCLDQL) + ZSNOWRIME(JL)
!
!      ENDIF
    ENDIF
  ENDDO
  
  ENDIF ! on IWARMRAIN > 1

  
  !----------------------------------------------------------------------
  ! 4.4a  MELTING OF SNOW and ICE
  !       with new implicit solver this also has to treat snow or ice
  !       precipitating from the level above... i.e. local ice AND flux.
  !       in situ ice and snow: could arise from LS advection or warming
  !       falling ice and snow: arrives by precipitation process
  !----------------------------------------------------------------------
  DO JL=KIDIA,KFDIA
    
    ZICETOT(JL)=ZQXFG(JL,NCLDQI)+ZQXFG(JL,NCLDQS)
    ZMELTMAX(JL) = 0.0

    ! If there are frozen hydrometeors present and dry-bulb temperature > 0degC
    IF(ZICETOT(JL) > ZEPSEC .AND. ZTP1(JL,JK) > RTT) THEN

      ! Calculate subsaturation
      ZSUBSAT = MAX(ZQSICE(JL,JK)-ZQX(JL,JK,NCLDQV),0.0)
      
      ! Calculate difference between dry-bulb (ZTP1) and the temperature 
      ! at which the wet-bulb=0degC (RTT-ZSUBSAT*....) using an approx.
      ! Melting only occurs if the wet-bulb temperature >0
      ! i.e. warming of ice particle due to melting > cooling 
      ! due to evaporation.
      ZTDMTW0 = ZTP1(JL,JK)-RTT-ZSUBSAT* &
                & (ZTW1+ZTW2*(PAP(JL,JK)-ZTW3)-ZTW4*(ZTP1(JL,JK)-ZTW5))
      ! Not implicit yet... 
      ! Ensure ZCONS1 is positive so that ZMELTMAX=0 if ZTDMTW0<0
      ZCONS1 = ABS(PTSPHY*(1.0+0.5*ZTDMTW0)/RTAUMEL)
      ZMELTMAX(JL) = MAX(ZTDMTW0*ZCONS1*ZRLDCP,0.0)
    ENDIF
  ENDDO

  ! Loop over frozen hydrometeors (ice, snow)
  DO JM=1,NCLV
   IF (IPHASE(JM) == 2) THEN
    JN = IMELT(JM)
    DO JL=KIDIA,KFDIA
      IF(ZMELTMAX(JL)>ZEPSEC .AND. ZICETOT(JL)>ZEPSEC) THEN
        ! Apply melting in same proportion as frozen hydrometeor fractions 
        ZALFA = ZQXFG(JL,JM)/ZICETOT(JL)
        ZMELT = MIN(ZQXFG(JL,JM),ZALFA*ZMELTMAX(JL))
        ! needed in first guess
        ! This implies that zqpretot has to be recalculated below
        ! since is not conserved here if ice falls and liquid doesn't
        ZQXFG(JL,JM)     = ZQXFG(JL,JM)-ZMELT
        ZQXFG(JL,JN)     = ZQXFG(JL,JN)+ZMELT
        ZSOLQA(JL,JN,JM) = ZSOLQA(JL,JN,JM)+ZMELT
        ZSOLQA(JL,JM,JN) = ZSOLQA(JL,JM,JN)-ZMELT
      ENDIF
    ENDDO
   ENDIF
  ENDDO
  
  !----------------------------------------------------------------------
  ! 4.4b  FREEZING of RAIN
  !----------------------------------------------------------------------
  DO JL=KIDIA,KFDIA 

    ! If rain present
    IF (ZQX(JL,JK,NCLDQR) > ZEPSEC) THEN

      IF (ZTP1(JL,JK) <= RTT .AND. ZTP1(JL,JK-1) > RTT) THEN
        ! Base of melting layer/top of refreezing layer so
        ! store rain/snow fraction for precip type diagnosis
        ! If mostly rain, then supercooled rain slow to freeze
        ! otherwise faster to freeze (snow or ice pellets)
        ZQPRETOT(JL) = MAX(ZQX(JL,JK,NCLDQS)+ZQX(JL,JK,NCLDQR),ZEPSEC)
        PRAINFRAC_TOPRFZ(JL) = ZQX(JL,JK,NCLDQR)/ZQPRETOT(JL)
        IF (PRAINFRAC_TOPRFZ(JL) > 0.8) THEN 
          LLRAINLIQ(JL) = .True.
        ELSE
          LLRAINLIQ(JL) = .False.
        ENDIF
      ENDIF
    
      ! If temperature less than zero
      IF (ZTP1(JL,JK) < RTT) THEN

        IF (LLRAINLIQ(JL)) THEN 

          ! Majority of raindrops completely melted
          ! Refreezing is by slow heterogeneous freezing
          
          ! Slope of rain particle size distribution
          ZLAMBDA = (RCL_FAC1/(ZRHO(JL)*ZQX(JL,JK,NCLDQR)))**RCL_FAC2

          ! Calculate freezing rate based on Bigg(1953) and Wisner(1972)
          ZTEMP = RCL_FZRAB * (ZTP1(JL,JK)-RTT)
          ZFRZ  = PTSPHY * (RCL_CONST5R/ZRHO(JL)) * (EXP(ZTEMP)-1.) &
                  & * ZLAMBDA**RCL_CONST6R
          ZFRZMAX(JL) = MAX(ZFRZ,0.0)

        ELSE

          ! Majority of raindrops only partially melted 
          ! Refreeze with a shorter timescale (reverse of melting...for now)
          
          ZCONS1 = ABS(PTSPHY*(1.0+0.5*(RTT-ZTP1(JL,JK)))/RTAUMEL)
          ZFRZMAX(JL) = MAX((RTT-ZTP1(JL,JK))*ZCONS1*ZRLDCP,0.0)

        ENDIF

        IF(ZFRZMAX(JL)>ZEPSEC) THEN
          ZFRZ = MIN(ZQX(JL,JK,NCLDQR),ZFRZMAX(JL))
          ZSOLQA(JL,NCLDQS,NCLDQR) = ZSOLQA(JL,NCLDQS,NCLDQR)+ZFRZ
          ZSOLQA(JL,NCLDQR,NCLDQS) = ZSOLQA(JL,NCLDQR,NCLDQS)-ZFRZ
        ENDIF
      ENDIF

    ENDIF

  ENDDO

  !----------------------------------------------------------------------
  ! 4.4c  FREEZING of LIQUID 
  !----------------------------------------------------------------------
  DO JL=KIDIA,KFDIA 
    ! not implicit yet... 
    ZFRZMAX(JL)=MAX((RTHOMO-ZTP1(JL,JK))*ZRLDCP,0.0)
  ENDDO

  JM = NCLDQL
  JN = IMELT(JM)
  DO JL=KIDIA,KFDIA
    IF(ZFRZMAX(JL)>ZEPSEC .AND. ZQXFG(JL,JM)>ZEPSEC) THEN
      ZFRZ = MIN(ZQXFG(JL,JM),ZFRZMAX(JL))
      ZSOLQA(JL,JN,JM) = ZSOLQA(JL,JN,JM)+ZFRZ
      ZSOLQA(JL,JM,JN) = ZSOLQA(JL,JM,JN)-ZFRZ
    ENDIF
  ENDDO

  !----------------------------------------------------------------------
  ! 4.5   EVAPORATION OF RAIN/SNOW
  !----------------------------------------------------------------------

  !----------------------------------------
  ! Rain evaporation scheme from Sundquist
  !----------------------------------------
 IF (IEVAPRAIN == 1) THEN

  ! Rain
  
  DO JL=KIDIA,KFDIA

    ZZRH=RPRECRHMAX+(1.0-RPRECRHMAX)*ZCOVPMAX(JL)/MAX(ZEPSEC,1.0-ZA(JL,JK))
    ZZRH=MIN(MAX(ZZRH,RPRECRHMAX),1.0)

    ZQE=(ZQX(JL,JK,NCLDQV)-ZA(JL,JK)*ZQSLIQ(JL,JK))/&
    & MAX(ZEPSEC,1.0-ZA(JL,JK))  
    !---------------------------------------------
    ! humidity in moistest ZCOVPCLR part of domain
    !---------------------------------------------
    ZQE=MAX(0.0,MIN(ZQE,ZQSLIQ(JL,JK)))
    LLO1=ZCOVPCLR(JL)>ZEPSEC .AND. &
       & ZQXFG(JL,NCLDQR)>ZEPSEC .AND. &
       & ZQE<ZZRH*ZQSLIQ(JL,JK)
    
    IF(LLO1) THEN
      ! note: zpreclr is a rain flux
      ZPRECLR = ZQXFG(JL,NCLDQR)*ZCOVPCLR(JL)/ &
       & SIGN(MAX(ABS(ZCOVPTOT(JL)*ZDTGDP(JL)),ZEPSILON),ZCOVPTOT(JL)*ZDTGDP(JL))

      !--------------------------------------
      ! actual microphysics formula in zbeta
      !--------------------------------------

      ZBETA1 = SQRT(PAP(JL,JK)/&
       & PAPH(JL,KLEV+1))/RVRFACTOR*ZPRECLR/&
       & MAX(ZCOVPCLR(JL),ZEPSEC)

      ZBETA=RG*RPECONS*0.5*ZBETA1**0.5777  

      ZDENOM  = 1.0+ZBETA*PTSPHY*ZCORQSLIQ(JL)
      ZDPR    = ZCOVPCLR(JL)*ZBETA*(ZQSLIQ(JL,JK)-ZQE)/ZDENOM*ZDP(JL)*ZRG_R
      ZDPEVAP = ZDPR*ZDTGDP(JL)

      !---------------------------------------------------------
      ! add evaporation term to explicit sink.
      ! this has to be explicit since if treated in the implicit
      ! term evaporation can not reduce rain to zero and model
      ! produces small amounts of rainfall everywhere. 
      !---------------------------------------------------------
      
      ! Evaporate rain
      ZEVAP = MIN(ZDPEVAP,ZQXFG(JL,NCLDQR))

      ZSOLQA(JL,NCLDQV,NCLDQR) = ZSOLQA(JL,NCLDQV,NCLDQR)+ZEVAP
      ZSOLQA(JL,NCLDQR,NCLDQV) = ZSOLQA(JL,NCLDQR,NCLDQV)-ZEVAP

      !-------------------------------------------------------------
      ! Reduce the total precip coverage proportional to evaporation
      ! to mimic the previous scheme which had a diagnostic
      ! 2-flux treatment, abandoned due to the new prognostic precip
      !-------------------------------------------------------------
      ZCOVPTOT(JL) = MAX(RCOVPMIN,ZCOVPTOT(JL)-MAX(0.0, &
       &            (ZCOVPTOT(JL)-ZA(JL,JK))*ZEVAP/ZQXFG(JL,NCLDQR)))

      ! Update fg field
      ZQXFG(JL,NCLDQR) = ZQXFG(JL,NCLDQR)-ZEVAP

    ENDIF
  ENDDO


 !---------------------------------------------------------
 ! Rain evaporation scheme based on Abel and Boutle (2013)
 !---------------------------------------------------------
 ELSEIF (IEVAPRAIN == 2) THEN

  DO JL=KIDIA,KFDIA

    !-----------------------------------------------------------------------
    ! Calculate relative humidity limit for rain evaporation 
    ! to avoid cloud formation and saturation of the grid box
    !-----------------------------------------------------------------------
    ! Limit RH for rain evaporation dependent on precipitation fraction 
    ZZRH=RPRECRHMAX+(1.0-RPRECRHMAX)*ZCOVPMAX(JL)/MAX(ZEPSEC,1.0-ZA(JL,JK))
    ZZRH=MIN(MAX(ZZRH,RPRECRHMAX),1.0)

    ! Critical relative humidity
    !ZRHC=RAMID
    !ZSIGK=PAP(JL,JK)/PAPH(JL,KLEV+1)
    ! Increase RHcrit to 1.0 towards the surface (eta>0.8)
    !IF(ZSIGK > 0.8) THEN
    !  ZRHC=RAMID+(1.0-RAMID)*((ZSIGK-0.8)/0.2)**2
    !ENDIF
    !ZZRH = MIN(ZRHC,ZZRH)

    ! Further limit RH for rain evaporation to 80_ (RHcrit in free troposphere)
    ZZRH = MIN(0.8,ZZRH)
  
    ZQE=MAX(0.0,MIN(ZQX(JL,JK,NCLDQV),ZQSLIQ(JL,JK)))

    LLO1=ZCOVPCLR(JL)>ZEPSEC .AND. &
       & ZQXFG(JL,NCLDQR)>ZEPSEC .AND. & 
       & ZQE<ZZRH*ZQSLIQ(JL,JK)

    IF(LLO1) THEN

      !-------------------------------------------
      ! Abel and Boutle (2012) evaporation
      !-------------------------------------------
      ! Calculate local precipitation (kg/kg)
      ZPRECLR = ZQXFG(JL,NCLDQR)/ZCOVPTOT(JL)

      ! Fallspeed air density correction 
      ZFALLCORR = (RDENSREF/ZRHO(JL))**0.4

      ! Saturation vapour pressure with respect to liquid phase
      ZESATLIQ = RV/RD*FOEELIQ(ZTP1(JL,JK))

      ! Slope of particle size distribution
      ZLAMBDA = (RCL_FAC1/(ZRHO(JL)*ZPRECLR))**RCL_FAC2 ! ZPRECLR=kg/kg

      ZEVAP_DENOM = RCL_CDENOM1*ZESATLIQ - RCL_CDENOM2*ZTP1(JL,JK)*ZESATLIQ &
              &+ RCL_CDENOM3*ZTP1(JL,JK)**3.*PAP(JL,JK)

      ! Temperature dependent conductivity
      ZCORR2= (ZTP1(JL,JK)/273.)**1.5*393./(ZTP1(JL,JK)+120.)
      ZKA = RCL_KA273*ZCORR2

      ZSUBSAT = MAX(ZZRH*ZQSLIQ(JL,JK)-ZQE,0.0)

      ZBETA = (0.5/ZQSLIQ(JL,JK))*ZTP1(JL,JK)**2.*ZESATLIQ* &
     & RCL_CONST1R*(ZCORR2/ZEVAP_DENOM)*(0.78/(ZLAMBDA**RCL_CONST4R)+ &
     & RCL_CONST2R*(ZRHO(JL)*ZFALLCORR)**0.5/ &
     & (ZCORR2**0.5*ZLAMBDA**RCL_CONST3R))
     
      ZDENOM  = 1.0+ZBETA*PTSPHY !*ZCORQSLIQ(JL)
      ZDPEVAP = ZCOVPCLR(JL)*ZBETA*PTSPHY*ZSUBSAT/ZDENOM

      !---------------------------------------------------------
      ! Add evaporation term to explicit sink.
      ! this has to be explicit since if treated in the implicit
      ! term evaporation can not reduce rain to zero and model
      ! produces small amounts of rainfall everywhere. 
      !---------------------------------------------------------
      
      ! Limit rain evaporation
      ZEVAP = MIN(ZDPEVAP,ZQXFG(JL,NCLDQR))

      ZSOLQA(JL,NCLDQV,NCLDQR) = ZSOLQA(JL,NCLDQV,NCLDQR)+ZEVAP
      ZSOLQA(JL,NCLDQR,NCLDQV) = ZSOLQA(JL,NCLDQR,NCLDQV)-ZEVAP

      !-------------------------------------------------------------
      ! Reduce the total precip coverage proportional to evaporation
      ! to mimic the previous scheme which had a diagnostic
      ! 2-flux treatment, abandoned due to the new prognostic precip
      !-------------------------------------------------------------
      ZCOVPTOT(JL) = MAX(RCOVPMIN,ZCOVPTOT(JL)-MAX(0.0, &
       &            (ZCOVPTOT(JL)-ZA(JL,JK))*ZEVAP/ZQXFG(JL,NCLDQR)))

      ! Update fg field 
      ZQXFG(JL,NCLDQR) = ZQXFG(JL,NCLDQR)-ZEVAP
    
    ENDIF
  ENDDO
  
ENDIF ! on IEVAPRAIN

  !----------------------------------------------------------------------
  ! 4.5   EVAPORATION OF SNOW
  !----------------------------------------------------------------------
  ! Snow
 IF (IEVAPSNOW == 1) THEN
  
  DO JL=KIDIA,KFDIA
    ZZRH=RPRECRHMAX+(1.0-RPRECRHMAX)*ZCOVPMAX(JL)/MAX(ZEPSEC,1.0-ZA(JL,JK))
    ZZRH=MIN(MAX(ZZRH,RPRECRHMAX),1.0)
    ZQE=(ZQX(JL,JK,NCLDQV)-ZA(JL,JK)*ZQSICE(JL,JK))/&
    & MAX(ZEPSEC,1.0-ZA(JL,JK))  

    !---------------------------------------------
    ! humidity in moistest ZCOVPCLR part of domain
    !---------------------------------------------
    ZQE=MAX(0.0,MIN(ZQE,ZQSICE(JL,JK)))
    LLO1=ZCOVPCLR(JL)>ZEPSEC .AND. &
       & ZQXFG(JL,NCLDQS)>ZEPSEC .AND. &
       & ZQE<ZZRH*ZQSICE(JL,JK)

    IF(LLO1) THEN
      ! note: zpreclr is a rain flux a
      ZPRECLR=ZQXFG(JL,NCLDQS)*ZCOVPCLR(JL)/ &
       & SIGN(MAX(ABS(ZCOVPTOT(JL)*ZDTGDP(JL)),ZEPSILON),ZCOVPTOT(JL)*ZDTGDP(JL))

      !--------------------------------------
      ! actual microphysics formula in zbeta
      !--------------------------------------

      ZBETA1=SQRT(PAP(JL,JK)/&
       & PAPH(JL,KLEV+1))/RVRFACTOR*ZPRECLR/&
       & MAX(ZCOVPCLR(JL),ZEPSEC)

      ZBETA=RG*RPECONS*(ZBETA1)**0.5777  

      ZDENOM=1.0+ZBETA*PTSPHY*ZCORQSICE(JL)
      ZDPR = ZCOVPCLR(JL)*ZBETA*(ZQSICE(JL,JK)-ZQE)/ZDENOM*ZDP(JL)*ZRG_R
      ZDPEVAP=ZDPR*ZDTGDP(JL)

      !---------------------------------------------------------
      ! add evaporation term to explicit sink.
      ! this has to be explicit since if treated in the implicit
      ! term evaporation can not reduce snow to zero and model
      ! produces small amounts of snowfall everywhere. 
      !---------------------------------------------------------
      
      ! Evaporate snow
      ZEVAP = MIN(ZDPEVAP,ZQXFG(JL,NCLDQS))

      ZSOLQA(JL,NCLDQV,NCLDQS) = ZSOLQA(JL,NCLDQV,NCLDQS)+ZEVAP
      ZSOLQA(JL,NCLDQS,NCLDQV) = ZSOLQA(JL,NCLDQS,NCLDQV)-ZEVAP
      
      !-------------------------------------------------------------
      ! Reduce the total precip coverage proportional to evaporation
      ! to mimic the previous scheme which had a diagnostic
      ! 2-flux treatment, abandoned due to the new prognostic precip
      !-------------------------------------------------------------
      ZCOVPTOT(JL) = MAX(RCOVPMIN,ZCOVPTOT(JL)-MAX(0.0, &
     &              (ZCOVPTOT(JL)-ZA(JL,JK))*ZEVAP/ZQXFG(JL,NCLDQS)))
      
      !Update first guess field
      ZQXFG(JL,NCLDQS) = ZQXFG(JL,NCLDQS)-ZEVAP

    ENDIF
  ENDDO
  !---------------------------------------------------------
  ELSEIF (IEVAPSNOW == 2) THEN

 
   DO JL=KIDIA,KFDIA

    !-----------------------------------------------------------------------
    ! Calculate relative humidity limit for snow evaporation 
    !-----------------------------------------------------------------------
    ZZRH=RPRECRHMAX+(1.0-RPRECRHMAX)*ZCOVPMAX(JL)/MAX(ZEPSEC,1.0-ZA(JL,JK))
    ZZRH=MIN(MAX(ZZRH,RPRECRHMAX),1.0)
    ZQE=(ZQX(JL,JK,NCLDQV)-ZA(JL,JK)*ZQSICE(JL,JK))/&
    & MAX(ZEPSEC,1.0-ZA(JL,JK))  
     
    !---------------------------------------------
    ! humidity in moistest ZCOVPCLR part of domain
    !---------------------------------------------
    ZQE=MAX(0.0,MIN(ZQE,ZQSICE(JL,JK)))
    LLO1=ZCOVPCLR(JL)>ZEPSEC .AND. &
       & ZQX(JL,JK,NCLDQS)>ZEPSEC .AND. &
       & ZQE<ZZRH*ZQSICE(JL,JK)

    IF(LLO1) THEN
      
      ! Calculate local precipitation (kg/kg)
      ZPRECLR = ZQX(JL,JK,NCLDQS)/ZCOVPTOT(JL)
      ZVPICE = FOEEICE(ZTP1(JL,JK))*RV/RD

      ! Particle size distribution
      ! ZTCG increases Ni with colder temperatures - essentially a 
      ! Fletcher or Meyers scheme? 
      ZTCG=1.0 !v1 EXP(RCL_X3I*(273.15-ZTP1(JL,JK))/8.18)
      ! ZFACX1I modification is based on Andrew Barrett's results
      ZFACX1S = 1.0 !v1 (ZICE0/1.E-5)**0.627

      ZAPLUSB   = RCL_APB1*ZVPICE-RCL_APB2*ZVPICE*ZTP1(JL,JK)+ &
     &             PAP(JL,JK)*RCL_APB3*ZTP1(JL,JK)**3
      ZCORRFAC  = (1.0/ZRHO(JL))**0.5
      ZCORRFAC2 = ((ZTP1(JL,JK)/273.0)**1.5)*(393.0/(ZTP1(JL,JK)+120.0))

      ZPR02 = ZRHO(JL)*ZPRECLR*RCL_CONST1S/(ZTCG*ZFACX1S)

      ZTERM1 = (ZQSICE(JL,JK)-ZQE)*ZTP1(JL,JK)**2*ZVPICE*ZCORRFAC2*ZTCG* &
     &          RCL_CONST2S*ZFACX1S/(ZRHO(JL)*ZAPLUSB*ZQSICE(JL,JK))
      ZTERM2 = 0.65*RCL_CONST6S*ZPR02**RCL_CONST4S+RCL_CONST3S*ZCORRFAC**0.5 &
     &          *ZRHO(JL)**0.5*ZPR02**RCL_CONST5S/ZCORRFAC2**0.5

      ZDPEVAP = MAX(ZCOVPCLR(JL)*ZTERM1*ZTERM2*PTSPHY,0.0)
 
      !--------------------------------------------------------------------
      ! Limit evaporation to snow amount
      !--------------------------------------------------------------------
      ZEVAP = MIN(ZDPEVAP,ZEVAPLIMICE(JL))
      ZEVAP = MIN(ZEVAP,ZQX(JL,JK,NCLDQS))

            
      ZSOLQA(JL,NCLDQV,NCLDQS) = ZSOLQA(JL,NCLDQV,NCLDQS)+ZEVAP
      ZSOLQA(JL,NCLDQS,NCLDQV) = ZSOLQA(JL,NCLDQS,NCLDQV)-ZEVAP
      
      !-------------------------------------------------------------
      ! Reduce the total precip coverage proportional to evaporation
      ! to mimic the previous scheme which had a diagnostic
      ! 2-flux treatment, abandoned due to the new prognostic precip
      !-------------------------------------------------------------
      ZCOVPTOT(JL) = MAX(RCOVPMIN,ZCOVPTOT(JL)-MAX(0.0, &
     &              (ZCOVPTOT(JL)-ZA(JL,JK))*ZEVAP/ZQX(JL,JK,NCLDQS)))
      
      !Update first guess field
      ZQXFG(JL,NCLDQS) = ZQXFG(JL,NCLDQS)-ZEVAP

    ENDIF    
  ENDDO
     
ENDIF ! on IEVAPSNOW

  !--------------------------------------
  ! Evaporate small precipitation amounts
  !--------------------------------------
  DO JM=1,NCLV
   IF (LLFALL(JM)) THEN 
    DO JL=KIDIA,KFDIA
      IF (ZQXFG(JL,JM)<RLMIN) THEN
        ZSOLQA(JL,NCLDQV,JM) = ZSOLQA(JL,NCLDQV,JM)+ZQXFG(JL,JM)
        ZSOLQA(JL,JM,NCLDQV) = ZSOLQA(JL,JM,NCLDQV)-ZQXFG(JL,JM)
      ENDIF
    ENDDO
   ENDIF
  ENDDO
  
  !######################################################################
  !            5.0  *** SOLVERS FOR A AND L ***
  ! now use an implicit solution rather than exact solution
  ! solver is forward in time, upstream difference for advection
  !######################################################################

  !---------------------------
  ! 5.1 solver for cloud cover
  !---------------------------
  DO JL=KIDIA,KFDIA
    ZANEW=(ZA(JL,JK)+ZSOLAC(JL))/(1.0+ZSOLAB(JL))
    ZANEW=MIN(ZANEW,1.0)
    IF (ZANEW<RAMIN) ZANEW=0.0
    ZDA(JL)=ZANEW-ZAORIG(JL,JK)
    !---------------------------------
    ! variables needed for next level
    !---------------------------------
    ZANEWM1(JL)=ZANEW
  ENDDO

  !--------------------------------
  ! 5.2 solver for the microphysics
  !--------------------------------

  !--------------------------------------------------------------
  ! Truncate explicit sinks to avoid negatives 
  ! Note: Species are treated in the order in which they run out
  ! since the clipping will alter the balance for the other vars
  !--------------------------------------------------------------

  DO JM=1,NCLV
    DO JN=1,NCLV
      DO JL=KIDIA,KFDIA
        LLINDEX3(JL,JN,JM)=.FALSE.
      ENDDO
    ENDDO
    DO JL=KIDIA,KFDIA
      ZSINKSUM(JL,JM)=0.0
    ENDDO
  ENDDO

  !----------------------------
  ! collect sink terms and mark
  !----------------------------
  DO JM=1,NCLV
    DO JN=1,NCLV
      DO JL=KIDIA,KFDIA
        ZSINKSUM(JL,JM)=ZSINKSUM(JL,JM)-ZSOLQA(JL,JM,JN) ! +ve total is bad
      ENDDO
    ENDDO
  ENDDO

  !---------------------------------------
  ! calculate overshoot and scaling factor
  !---------------------------------------
  DO JM=1,NCLV
    DO JL=KIDIA,KFDIA
      ZMAX=MAX(ZQX(JL,JK,JM),ZEPSEC)
      ZRAT=MAX(ZSINKSUM(JL,JM),ZMAX)
      ZRATIO(JL,JM)=ZMAX/ZRAT
    ENDDO
  ENDDO
  !--------------------------------------------------------
  ! now sort zratio to find out which species run out first
  !--------------------------------------------------------
  DO JM=1,NCLV
    DO JL=KIDIA,KFDIA
      IORDER(JL,JM)=-999
    ENDDO
  ENDDO
  DO JN=1,NCLV
    DO JL=KIDIA,KFDIA
      LLINDEX1(JL,JN)=.TRUE.
    ENDDO
  ENDDO
  DO JM=1,NCLV
    DO JL=KIDIA,KFDIA
      ZMIN(JL)=1.E32
    ENDDO
    DO JN=1,NCLV
      DO JL=KIDIA,KFDIA
        IF (LLINDEX1(JL,JN) .AND. ZRATIO(JL,JN)<ZMIN(JL)) THEN
          IORDER(JL,JM)=JN
          ZMIN(JL)=ZRATIO(JL,JN)
        ENDIF
      ENDDO
    ENDDO
    DO JL=KIDIA,KFDIA
      JO=IORDER(JL,JM)
      LLINDEX1(JL,JO)=.FALSE. ! marked as searched
    ENDDO
  ENDDO

  !--------------------------------------------
  ! scale the sink terms, in the correct order, 
  ! recalculating the scale factor each time
  !--------------------------------------------
  DO JM=1,NCLV
    DO JL=KIDIA,KFDIA
      ZSINKSUM(JL,JM)=0.0
    ENDDO
  ENDDO

  !----------------
  ! recalculate sum
  !----------------
  DO JM=1,NCLV
!   DO JN=1,NCLV
    DO JL=KIDIA,KFDIA
      JO=IORDER(JL,JM)
!     ZZSUM=ZSINKSUM(JL,JO)
!DIR$ IVDEP
!DIR$ PREFERVECTOR
      DO JN=1,NCLV
        LLINDEX3(JL,JO,JN)=ZSOLQA(JL,JO,JN)<0.0
!       ZSINKSUM(JL,JO)=ZSINKSUM(JL,JO)-ZSOLQA(JL,JO,JN)
!       ZZSUM=ZZSUM-ZSOLQA(JL,JO,JN)
      ENDDO
      ZSINKSUM(JL,JO)=ZSINKSUM(JL,JO)-SUM(ZSOLQA(JL,JO,1:NCLV))
    ENDDO
    !---------------------------
    ! recalculate scaling factor
    !---------------------------
    DO JL=KIDIA,KFDIA
      JO=IORDER(JL,JM)
      ZMM=MAX(ZQX(JL,JK,JO),ZEPSEC)
      ZRR=MAX(ZSINKSUM(JL,JO),ZMM)
      ZRATIO(JL,JO)=ZMM/ZRR
    ENDDO
    !------
    ! scale
    !------
    DO JL=KIDIA,KFDIA
      JO=IORDER(JL,JM)
      ZZRATIO=ZRATIO(JL,JO)
!DIR$ IVDEP
!DIR$ PREFERVECTOR
      DO JN=1,NCLV
        IF (LLINDEX3(JL,JO,JN)) THEN
          ZSOLQA(JL,JO,JN)=ZSOLQA(JL,JO,JN)*ZZRATIO
          ZSOLQA(JL,JN,JO)=ZSOLQA(JL,JN,JO)*ZZRATIO
        ENDIF
      ENDDO
    ENDDO
  ENDDO

  !--------------------------------------------------------------
  ! 5.2.2 Solver
  !------------------------

  !------------------------
  ! set the LHS of equation  
  !------------------------
  DO JM=1,NCLV
    DO JN=1,NCLV
      !----------------------------------------------
      ! diagonals: microphysical sink terms+transport
      !----------------------------------------------
      IF (JN==JM) THEN
        DO JL=KIDIA,KFDIA
          ZQLHS(JL,JN,JM)=1.0 + ZFALLSINK(JL,JM)
          DO JO=1,NCLV
            ZQLHS(JL,JN,JM)=ZQLHS(JL,JN,JM) + ZSOLQB(JL,JO,JN)
          ENDDO
        ENDDO
      !------------------------------------------
      ! non-diagonals: microphysical source terms
      !------------------------------------------
      ELSE
        DO JL=KIDIA,KFDIA
         ZQLHS(JL,JN,JM)= -ZSOLQB(JL,JN,JM) ! here is the delta T - missing from doc.
        ENDDO
      ENDIF    
    ENDDO
  ENDDO

  !------------------------
  ! set the RHS of equation  
  !------------------------
  DO JM=1,NCLV
    DO JL=KIDIA,KFDIA
      !---------------------------------
      ! sum the explicit source and sink
      !---------------------------------
      ZEXPLICIT=0.0
      DO JN=1,NCLV
        ZEXPLICIT=ZEXPLICIT+ZSOLQA(JL,JM,JN) ! sum over middle index
      ENDDO
      ZQXN(JL,JM)=ZQX(JL,JK,JM)+ZEXPLICIT
    ENDDO
  ENDDO

  !-----------------------------------
  ! *** solve by LU decomposition: ***
  !-----------------------------------

  ! Note: This fast way of solving NCLVxNCLV system
  !       assumes a good behaviour (i.e. non-zero diagonal
  !       terms with comparable orders) of the matrix stored
  !       in ZQLHS. For the moment this is the case but
  !       be aware to preserve it when doing eventual 
  !       modifications.

  ! Non pivoting recursive factorization 
  DO JN = 1, NCLV-1  ! number of steps
    DO JM = JN+1,NCLV ! row index
      ZQLHS(KIDIA:KFDIA,JM,JN)=ZQLHS(KIDIA:KFDIA,JM,JN) &
       &                     / ZQLHS(KIDIA:KFDIA,JN,JN)
      DO IK=JN+1,NCLV ! column index
        DO JL=KIDIA,KFDIA
          ZQLHS(JL,JM,IK)=ZQLHS(JL,JM,IK)-ZQLHS(JL,JM,JN)*ZQLHS(JL,JN,IK)
        ENDDO
      ENDDO
    ENDDO
  ENDDO        

  ! Backsubstitution 
  !  step 1 
  DO JN=2,NCLV
    DO JM = 1,JN-1
      ZQXN(KIDIA:KFDIA,JN)=ZQXN(KIDIA:KFDIA,JN)-ZQLHS(KIDIA:KFDIA,JN,JM) &
       &  *ZQXN(KIDIA:KFDIA,JM)
    ENDDO
  ENDDO
  !  step 2
  ZQXN(KIDIA:KFDIA,NCLV)=ZQXN(KIDIA:KFDIA,NCLV)/ZQLHS(KIDIA:KFDIA,NCLV,NCLV)
  DO JN=NCLV-1,1,-1
    DO JM = JN+1,NCLV
      ZQXN(KIDIA:KFDIA,JN)=ZQXN(KIDIA:KFDIA,JN)-ZQLHS(KIDIA:KFDIA,JN,JM) &
       &  *ZQXN(KIDIA:KFDIA,JM)
    ENDDO
    ZQXN(KIDIA:KFDIA,JN)=ZQXN(KIDIA:KFDIA,JN)/ZQLHS(KIDIA:KFDIA,JN,JN)
  ENDDO

  ! Ensure no small values (including negatives) remain in cloud variables nor
  ! precipitation rates.
  ! Evaporate l,i,r,s to water vapour. Latent heating taken into account below
  DO JN=1,NCLV-1
    DO JL=KIDIA,KFDIA
      IF (ZQXN(JL,JN) < ZEPSEC) THEN
        ZQXN(JL,NCLDQV) = ZQXN(JL,NCLDQV)+ZQXN(JL,JN)
        ZQXN(JL,JN)     = 0.0
      ENDIF
    ENDDO
  ENDDO

  !--------------------------------
  ! variables needed for next level
  !--------------------------------
  DO JM=1,NCLV
    DO JL=KIDIA,KFDIA
      ZQXNM1(JL,JM)    = ZQXN(JL,JM)
      ZQXN2D(JL,JK,JM) = ZQXN(JL,JM)
    ENDDO
  ENDDO

  !------------------------------------------------------------------------
  ! 5.3 Precipitation/sedimentation fluxes to next level
  !     diagnostic precipitation fluxes
  !     It is this scaled flux that must be used for source to next layer
  !------------------------------------------------------------------------

  DO JM=1,NCLV
    DO JL=KIDIA,KFDIA
      ZPFPLSX(JL,JK+1,JM) = ZFALLSINK(JL,JM)*ZQXN(JL,JM)*ZRDTGDP(JL)
    ENDDO
  ENDDO

  ! Ensure precipitation fraction is zero if no precipitation
  DO JL=KIDIA,KFDIA
    ZQPRETOT(JL) =ZPFPLSX(JL,JK+1,NCLDQS)+ZPFPLSX(JL,JK+1,NCLDQR)
  ENDDO
  DO JL=KIDIA,KFDIA
    IF (ZQPRETOT(JL)<ZEPSEC) THEN
      ZCOVPTOT(JL)=0.0
    ENDIF
  ENDDO
  
  !######################################################################
  !              6  *** UPDATE TENDANCIES ***
  !######################################################################

  !--------------------------------
  ! 6.1 Temperature and CLV budgets 
  !--------------------------------

  DO JM=1,NCLV-1
    DO JL=KIDIA,KFDIA

      ! calculate fluxes in and out of box for conservation of TL
      ZFLUXQ(JL,JM)=ZPSUPSATSRCE(JL,JM)+ZCONVSRCE(JL,JM)+ZFALLSRCE(JL,JM)-&
                    & (ZFALLSINK(JL,JM)+ZCONVSINK(JL,JM))*ZQXN(JL,JM)
    ENDDO

    IF (IPHASE(JM)==1) THEN
      DO JL=KIDIA,KFDIA
        tendency_loc_T(JL,JK)=tendency_loc_T(JL,JK)+ &
          & RALVDCP*(ZQXN(JL,JM)-ZQX(JL,JK,JM)-ZFLUXQ(JL,JM))*ZQTMST
      ENDDO
    ENDIF

    IF (IPHASE(JM)==2) THEN
      DO JL=KIDIA,KFDIA
        tendency_loc_T(JL,JK)=tendency_loc_T(JL,JK)+ &
          & RALSDCP*(ZQXN(JL,JM)-ZQX(JL,JK,JM)-ZFLUXQ(JL,JM))*ZQTMST
      ENDDO
    ENDIF

      !----------------------------------------------------------------------
      ! New prognostic tendencies - ice,liquid rain,snow 
      ! Note: CLV arrays use PCLV in calculation of tendency while humidity
      !       uses ZQX. This is due to clipping at start of cloudsc which
      !       include the tendency already in tendency_loc_T and tendency_loc_q. ZQX was reset
      !----------------------------------------------------------------------
    DO JL=KIDIA,KFDIA
      tendency_loc_cld(JL,JK,JM)=tendency_loc_cld(JL,JK,JM)+(ZQXN(JL,JM)-ZQX0(JL,JK,JM))*ZQTMST
    ENDDO

  ENDDO

  DO JL=KIDIA,KFDIA
    !----------------------
    ! 6.2 Humidity budget
    !----------------------
    tendency_loc_q(JL,JK)=tendency_loc_q(JL,JK)+(ZQXN(JL,NCLDQV)-ZQX(JL,JK,NCLDQV))*ZQTMST

    !-------------------
    ! 6.3 cloud cover 
    !-----------------------
    tendency_loc_a(JL,JK)=tendency_loc_a(JL,JK)+ZDA(JL)*ZQTMST
  ENDDO
 
!--------------------------------------------------
! Copy precipitation fraction into output variable
!-------------------------------------------------
  DO JL=KIDIA,KFDIA
    PCOVPTOT(JL,JK) = ZCOVPTOT(JL)
  ENDDO
 
ENDDO ! on vertical level JK

!----------------------------------------------------------------------
!                       END OF VERTICAL LOOP
!----------------------------------------------------------------------

!######################################################################
!              8  *** FLUX/DIAGNOSTICS COMPUTATIONS ***
!######################################################################

!--------------------------------------------------------------------
! Copy general precip arrays back into PFP arrays for GRIB archiving
! Add rain and liquid fluxes, ice and snow fluxes
!--------------------------------------------------------------------
DO JK=1,KLEV+1
  DO JL=KIDIA,KFDIA
    PFPLSL(JL,JK) = ZPFPLSX(JL,JK,NCLDQR)+ZPFPLSX(JL,JK,NCLDQL)
    PFPLSN(JL,JK) = ZPFPLSX(JL,JK,NCLDQS)+ZPFPLSX(JL,JK,NCLDQI)
  ENDDO
ENDDO

!--------
! Fluxes:
!--------
DO JL=KIDIA,KFDIA
  PFSQLF(JL,1)  = 0.0
  PFSQIF(JL,1)  = 0.0
  PFSQRF(JL,1)  = 0.0
  PFSQSF(JL,1)  = 0.0
  PFCQLNG(JL,1) = 0.0
  PFCQNNG(JL,1) = 0.0
  PFCQRNG(JL,1) = 0.0 !rain
  PFCQSNG(JL,1) = 0.0 !snow
! fluxes due to turbulence
  PFSQLTUR(JL,1) = 0.0
  PFSQITUR(JL,1) = 0.0
ENDDO

DO JK=1,KLEV
  DO JL=KIDIA,KFDIA

    ZGDPH_R = -ZRG_R*(PAPH(JL,JK+1)-PAPH(JL,JK))*ZQTMST
    PFSQLF(JL,JK+1)  = PFSQLF(JL,JK)
    PFSQIF(JL,JK+1)  = PFSQIF(JL,JK)
    PFSQRF(JL,JK+1)  = PFSQLF(JL,JK)
    PFSQSF(JL,JK+1)  = PFSQIF(JL,JK)
    PFCQLNG(JL,JK+1) = PFCQLNG(JL,JK)
    PFCQNNG(JL,JK+1) = PFCQNNG(JL,JK)
    PFCQRNG(JL,JK+1) = PFCQLNG(JL,JK)
    PFCQSNG(JL,JK+1) = PFCQNNG(JL,JK)
    PFSQLTUR(JL,JK+1) = PFSQLTUR(JL,JK)
    PFSQITUR(JL,JK+1) = PFSQITUR(JL,JK)

    ZALFAW=ZFOEALFA(JL,JK)

    ! Liquid , LS scheme minus detrainment
    PFSQLF(JL,JK+1)=PFSQLF(JL,JK+1)+ &
     &(ZQXN2D(JL,JK,NCLDQL)-ZQX0(JL,JK,NCLDQL)+PVFL(JL,JK)*PTSPHY-ZALFAW*PLUDE(JL,JK))*ZGDPH_R
    ! liquid, negative numbers
    PFCQLNG(JL,JK+1)=PFCQLNG(JL,JK+1)+ZLNEG(JL,JK,NCLDQL)*ZGDPH_R

    ! liquid, vertical diffusion
    PFSQLTUR(JL,JK+1)=PFSQLTUR(JL,JK+1)+PVFL(JL,JK)*PTSPHY*ZGDPH_R

    ! Rain, LS scheme 
    PFSQRF(JL,JK+1)=PFSQRF(JL,JK+1)+(ZQXN2D(JL,JK,NCLDQR)-ZQX0(JL,JK,NCLDQR))*ZGDPH_R 
    ! rain, negative numbers
    PFCQRNG(JL,JK+1)=PFCQRNG(JL,JK+1)+ZLNEG(JL,JK,NCLDQR)*ZGDPH_R

    ! Ice , LS scheme minus detrainment
    PFSQIF(JL,JK+1)=PFSQIF(JL,JK+1)+ &
     & (ZQXN2D(JL,JK,NCLDQI)-ZQX0(JL,JK,NCLDQI)+PVFI(JL,JK)*PTSPHY-(1.0-ZALFAW)*PLUDE(JL,JK))*ZGDPH_R
     ! ice, negative numbers
    PFCQNNG(JL,JK+1)=PFCQNNG(JL,JK+1)+ZLNEG(JL,JK,NCLDQI)*ZGDPH_R

    ! ice, vertical diffusion
    PFSQITUR(JL,JK+1)=PFSQITUR(JL,JK+1)+PVFI(JL,JK)*PTSPHY*ZGDPH_R

    ! snow, LS scheme
    PFSQSF(JL,JK+1)=PFSQSF(JL,JK+1)+(ZQXN2D(JL,JK,NCLDQS)-ZQX0(JL,JK,NCLDQS))*ZGDPH_R 
    ! snow, negative numbers
    PFCQSNG(JL,JK+1)=PFCQSNG(JL,JK+1)+ZLNEG(JL,JK,NCLDQS)*ZGDPH_R
  ENDDO
ENDDO

!-----------------------------------
! enthalpy flux due to precipitation
!-----------------------------------
DO JK=1,KLEV+1
  DO JL=KIDIA,KFDIA
    PFHPSL(JL,JK) = -RLVTT*PFPLSL(JL,JK)
    PFHPSN(JL,JK) = -RLSTT*PFPLSN(JL,JK)
  ENDDO
ENDDO

!===============================================================================
!END ASSOCIATE
!IF (LHOOK) CALL DR_HOOK('CLOUDSC',1,ZHOOK_HANDLE)
END SUBROUTINE CLOUDSC


