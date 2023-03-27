PROGRAM mwe_test
    INTEGER, PARAMETER  :: KLON = 100
    INTEGER, PARAMETER  :: KLEV = 100
    INTEGER, PARAMETER  :: NCLV = 100

    INTEGER :: KIDIA 
    INTEGER :: KFDIA
    INTEGER :: NCLDQV


    REAL :: ZQX(KLON,KLEV,NCLV)

    REAL :: ZFOKOOP(KLON)
    REAL :: ZSUPSAT(KLON)

    CALL mwe_test_routine(&
        & KLON, KLEV, KIDIA, KFDIA, NCLV,  NCLDQV,  &
        &      ZQX,   &
        &  ZFOKOOP,  ZSUPSAT)
END

SUBROUTINE mwe_test_routine(&
        & KLON, KLEV, KIDIA, KFDIA, NCLV,  NCLDQV,  &
        &      ZQX,   &
        &  ZFOKOOP,  ZSUPSAT)

    INTEGER  :: KLON
    INTEGER  :: KLEV
    INTEGER  :: KIDIA 
    INTEGER  :: KFDIA
    INTEGER  :: NCLV
    INTEGER  :: NCLDQV


    REAL :: ZQX(KLON,KLEV,NCLV)

    REAL :: ZFOKOOP(KLON)
    REAL :: ZSUPSAT(KLON)

    REAL :: ZFAC

    INTEGER :: JK, JL

    DO JK=KIDIA,KFDIA

        DO JL=KIDIA,KFDIA
            ZFOKOOP(JL)=JL*JK
        ENDDO

        DO JL=KIDIA,KFDIA
    
            ZFAC  = ZFOKOOP(JL)
            ZSUPSAT(JL) = (ZQX(JL,JK,NCLDQV)-ZFAC)
    
        ENDDO ! on JL

    ENDDO

END SUBROUTINE mwe_test_routine
