import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import rpy2.situation
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
# import rpy2.interactive as r
#
# r.packages.utils.install_packages("mvoutlier")
# r.packages.utils.install_packages("extremevalues")

utils = importr("utils")
base = importr('base')
#extremevalues = importr('extremevalues')
outliers = importr('outliers')

class OutlierDetection:
    '''
    To detect outliers with R packages,mvoutlier and extremevalues
    '''
    def __init__(self):
        pass


    def detect_outlier_with_extremevalues(self, df, col):
        '''
        :param df: dataframe
        :param col: str, to detect the column weather has outlier
        :param alpha: uni.plot function mvoutlier 
        :return: 
        '''
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_from_pd_df = ro.conversion.py2rpy(df)

        with localconverter(ro.default_converter + pandas2ri.converter):
            pd_from_r_df = ro.conversion.rpy2py(r_from_pd_df)

        rpy2.robjects.r('''
            getOutliersII <- function(y, alpha=c(0.05, 0.05), FLim=c(0.1, 0.9), distribution="normal", returnResiduals=TRUE)
            {
            # Input check
               if ( !is.vector(y) ) 
                  stop("First argument is not of type vector")
               if ( sum(y < 0) > 0 & !(distribution == "normal") )
                  stop("First argument contains nonpositive values")
               if ( sum( alpha <= 0 | alpha >= 1, na.rm=TRUE ) > 0 )
                  stop("Values of alpha must be between 0 and 1")
               if ( FLim[2] <= FLim[1] | sum( FLim < 0 | FLim > 1) >0 )
                  stop("Invalid range in FLim: 0<=FLim[1]<FLim[2]<=1")
               if ( ! distribution %in% c("lognormal", "pareto", "exponential", "weibull", "normal") )
                  stop("Invalid distribution (lognormal, pareto, exponential, weibull, normal).")

            # prepare for regression
               iy <- order(y) 
               Y <- y[iy]
               N <- length(Y)
               p <- seq(1,N)/(N+1)
               iLambda <- which( p >= FLim[1] & p <= FLim[2] )
               if (length(iLambda) <= 2 )
                  stop("Number of observations in fit is too small to estimate variance of residuals (need at least 3)")
            # regression and limit calculation
               par <- switch(distribution,
                  lognormal   = qqLognormalLimit(Y, p, iLambda, alpha),
                  weibull     = qqWeibullLimit(Y, p, iLambda, alpha),
                  pareto      = qqParetoLimit(Y, p, iLambda, alpha),
                  exponential = qqExponentialLimit(Y, p, iLambda, alpha),
                  normal      = qqNormalLimit(Y, p, iLambda, alpha),
                  )
            
            # locate outliers
               iLmin <- iLplus <- numeric(0)
               i <- N + 1
               while ( par$residuals[i-1] > par$limit[2] & i > tail(iLambda,1)+1 )
                  i <- i-1
               if ( i <= N )
                  iLplus <- iy[i:N]
            
               i <- 0
               while ( par$residuals[i+1] < par$limit[1] & i < iLambda[1]-1 )
                  i <- i+1
               if ( i > 0 )
                  iLmin <- iy[1:i]
            
            # organize output
               out <- par
               out$method <- "Method II"
               out$distribution <- distribution
               out$iRight <- iLplus
               out$iLeft <- iLmin 
               out$nOut <- c(Left=length(iLmin),Right=length(iLplus))
               out$yMin <- head(Y[iLambda],1)
               out$yMax <- tail(Y[iLambda],1)
               out$alphaConf<-c(Left=alpha[1], Right=alpha[2])
               out$nFit <- length(iLambda)
               if ( returnResiduals ){
                  out$residuals <- numeric(N)
                  out$residuals[iy] <- par$residuals
                  }
               return(out)
            }
            
            
            getOutliersI <- function(y, rho=c(1,1), FLim=c(0.1,0.9), distribution="normal")
            {
            
               if ( !is.vector(y) ) 
                  stop("First argument is not of type vector")
               if ( sum(y < 0) > 0 & !(distribution == "normal") )
                  stop("First argument contains nonpositive values")
               if ( sum( rho <= 0, na.rm=TRUE ) > 0 )
                  stop("Values of rho must be positive")
               if ( FLim[2] <= FLim[1] | sum( FLim < 0 | FLim > 1) >0 )
                  stop("Invalid range in FLim: 0<=FLim[1]<FLim[2]<=1")
               if ( ! distribution %in% c("lognormal", "pareto", "exponential", "weibull", "normal") )
                  stop("Invalid distribution (lognormal, pareto, exponential, weibull, normal).")
            
               Y <- y;
             
               y <- sort(y);
               N <- length(y)
               P <- (1:N)/(N+1)
               Lambda <- P >= FLim[1] & P<=FLim[2]
             
               y <- y[Lambda];
               p <- P[Lambda];
               out <- switch(distribution,
                     lognormal = getLognormalLimit(y, p, N, rho),
                     pareto = getParetoLimit(y, p, N, rho),
                     exponential = getExponentialLimit(y, p, N, rho),
                     weibull = getWeibullLimit(y, p, N, rho),
                     normal = getNormalLimit(y, p, N, rho)
                     )
               
               out$method <- "Method I"
               out$distribution=distribution
               out$iRight = which( Y > out$limit[2] )
               out$iLeft = which( Y < out$limit[1] )
               out$nOut = c(Left=length(out$iLeft), Right=length(out$iRight))
               out$yMin <- y[1]
               out$yMax <- tail(y,1)
               out$rho = c(Left=rho[1], Right=rho[2])
               out$Fmin = FLim[1]
               out$Fmax = FLim[2]
            
               return(out);
            }
            
            
            getOutliers <- function(y, method="I",  ...)
            {
                if ( !(method %in% c("I","II") ) )
                    stop("method not recognized (choose I or II)")
                out <- switch( method,
                    I = getOutliersI(y, ...),
                    II = getOutliersII(y, ...)
                    )
                return(out)
            }
            
            getNormalLimit <- function(y, p, N, rho)
            {
               param <- fitNormal(y,p)
               ell <- c(Left=-Inf, Right=Inf)
               if ( !is.na(rho[1]) )
                  ell[1] <- sqrt(2)*param$sigma*invErf(2*rho[1]/N-1)+param$mu
               if ( !is.na(rho[2]) )
                  ell[2] <- sqrt(2)*param$sigma*invErf(1-2*rho[2]/N)+param$mu
               return(list(mu=param$mu, 
                           sigma=param$sigma,
                           nFit=length(y),
                           R2=param$R2,
                           limit=ell)
                     )
            }
            fitNormal <- function(y, p)
            {
               if ( !is.vector(y) ) 
                  stop("First argument is not of type vector")
               if ( !is.vector(p)) 
                  stop("First argument is not of type vector")
               if ( sum(p<=0) > 0 | sum(p>=1) >0 )
                  stop("Second argument contains values out of range (0,1)")
               if (length(y) != length(p))
                  stop("First and second argument have different length");
            
               N <- length(y);
               Y <- as.matrix(y,nrow=N)
               p <- as.matrix(p,nrow=N)
            
            
               A <- matrix(0,nrow=N,ncol=2)
               A[,1] <- 1+double(N);
               A[,2] <- sqrt(2)*invErf(2*p-1)
               param <- solve(t(A) %*% A) %*% t(A) %*% Y
               r2 <- 1 - var(A%*%param - y)/var(y);
               return(list(mu=param[1], sigma=param[2], R2=r2));
            }
            invErf <- function(x)
            {
                if ( sum(x >= 1) > 0  | sum(x <= -1) > 0 )
                    stop("Argument must be between -1 and 1")
            
                return(qnorm((1+x)/2)/sqrt(2));
            
            }

            detect_outlier_with_extremevalues <- function(df,s){
                index <- 1:dim(df)[1]
                cols <- unlist(strsplit(s, split = "-"))
                for (column in cols){
                    p1 <- getOutliers(df[,column],method='I')
                    outlier_index <- as.vector(which(p1$outliers == T))
                    col_outlier <- sample(c(0), dim(df)[1], replace=T)
                    col_outlier[outlier_index] <- 1
                    new_col <- paste(column, 'outlier', sep = "_")
                    df[,new_col] <- col_outlier
                    }
                return(df)
                }
                ''')

        rf = rpy2.robjects.r['detect_outlier_with_extremevalues']
        with localconverter(ro.default_converter + pandas2ri.converter):
            data = ro.conversion.rpy2py(rf(r_from_pd_df,col))
        return data

    def detect_outlier_with_outliers(self, df, col):
        '''
        :param df: 
        :param col: str, to detect the column weather has outlier
        :param method: getOutliers is a wrapper function for getOutliersI and getOutliersII.
        :return: 
        '''
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_from_pd_df = ro.conversion.py2rpy(df)

        with localconverter(ro.default_converter + pandas2ri.converter):
            pd_from_r_df = ro.conversion.rpy2py(r_from_pd_df)

        rpy2.robjects.r('''
            detect_outlier_with_outliers <- function(df,s){
                index <- 1:dim(df)[1]
                cols <- unlist(strsplit(s, split = "-"))
                for (column in cols){
                    p1 <- outlier(df[,column],opposite=TRUE)
                    outlier_index <- as.vector(which(df[,column] == p1))
                    col_outlier <- sample(c(0), dim(df)[1], replace=T)
                    col_outlier[outlier_index] <- 1
                    new_col <- paste(column, 'outlier', sep = "_")
                    df[,new_col] <- col_outlier
                    }
                return(df)
                }
                    ''')

        rf = rpy2.robjects.r['detect_outlier_with_outliers']
        with localconverter(ro.default_converter + pandas2ri.converter):
            data = ro.conversion.rpy2py(rf(r_from_pd_df,col))
        return data

if __name__ == '__main__':
    df = pd.read_csv('data.csv')
    print(OutlierDetection().detect_outlier_with_extremevalues(df,'V1-V3-V4-V5-V7-V9'))
    print(OutlierDetection().detect_outlier_with_outliers(df,'V1-V3-V4-V5-V7-V9'))