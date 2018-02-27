output: html\_document
======================

BTC Modeling
------------

This is an attempt to model the BTC price using previous days data.

Retrieving BTC data
-------------------

We use the Quandl API to retrieve historical BTC prices, filter &gt;=
2014

    btc <- Quandl("BCHARTS/BTCDEEUR")
    btc <- subset(btc, Date>='2014-01-01')
    dim(btc)

    ## [1] 1518    8

BTC trends
----------

the daily weighted BTC price

![](BTC_Test_files/figure-markdown_strict/chart-1.png)

Dataset correlation coefficients
--------------------------------

a simple correlationchart, showing the high degree of correlation
between intra-day data

![](BTC_Test_files/figure-markdown_strict/lm.simple-1.png)

Data Preparation
================

### Preparing prediction using historical data

historical data via lag

    ds <- btc %>%
        mutate(Price = `Weighted Price`) %>%
        dplyr::select(Date, Price) %>%
        arrange(Date) %>%
        mutate(Grow_0 = Price > lag(Price, 1), 
               Grow_1 = lag(Price, 1)>lag(Price, 2), 
               Grow_2 = lag(Price, 2)>lag(Price, 3), 
               Grow_3 = lag(Price, 3)>lag(Price, 4), 
               Grow_4 = lag(Price, 4)>lag(Price, 5), 
               Grow_5 = lag(Price, 5)>lag(Price, 6), 
               Grow_6 = lag(Price, 6)>lag(Price, 7), 
               Grow_5 = lag(Price, 7)>lag(Price, 8), 
               Grow_1w = lag(Price, 1)>lag(Price, 8), 
               Grow_2w = lag(Price, 1)>lag(Price, 15)) %>%
        dplyr::select(-Date, - Price)
    rownames(ds) <- btc$Date

      ds <- ds[complete.cases(ds),]
      head(ds)

    ##            Grow_0 Grow_1 Grow_2 Grow_3 Grow_4 Grow_5 Grow_6 Grow_1w
    ## 2018-02-11  FALSE   TRUE  FALSE  FALSE  FALSE   TRUE  FALSE    TRUE
    ## 2018-02-10  FALSE  FALSE   TRUE  FALSE  FALSE  FALSE   TRUE   FALSE
    ## 2018-02-09   TRUE  FALSE  FALSE   TRUE  FALSE   TRUE  FALSE   FALSE
    ## 2018-02-08   TRUE   TRUE  FALSE  FALSE   TRUE  FALSE  FALSE   FALSE
    ## 2018-02-07   TRUE   TRUE   TRUE  FALSE  FALSE  FALSE  FALSE   FALSE
    ## 2018-02-06  FALSE   TRUE   TRUE   TRUE  FALSE  FALSE   TRUE    TRUE
    ##            Grow_2w
    ## 2018-02-11    TRUE
    ## 2018-02-10    TRUE
    ## 2018-02-09    TRUE
    ## 2018-02-08   FALSE
    ## 2018-02-07   FALSE
    ## 2018-02-06   FALSE

      x = ds[, -1]
      y = as.factor(ds$Grow_0)

Prediction comparison
=====================

### BTC Modeling using several ML algos

a logistic regression model to see how today's trend can be predicted
based on historical data

Tools: 4 popular ML method x 30 results ( 3 repeats of 10-fold cross
validation)

Methods: RF, lvw, svm, gbm

    control = trainControl(method='cv',number=10, repeats=3)

    ## Warning: `repeats` has no meaning for this resampling method.

    set.seed(123)
    model.rf  = train(x,y,model='rf' ,trControl=control, metric="Accuracy")
    model.lvq = train(x,y,model='lvq',trControl=control, metric="Accuracy")
    model.gbm = train(x,y,model='gbm',trControl=control, metric="Accuracy", verbose=FALSE)
    model.svm = train(x,y,model='svm',trControl=control, metric="Accuracy")

    # collect resamples
    results <- resamples(list(RF = model.rf, LVQ=model.lvq, GBM=model.gbm, SVM=model.svm))
    # summarize the distributions
    summary(results)

    ## 
    ## Call:
    ## summary.resamples(object = results)
    ## 
    ## Models: RF, LVQ, GBM, SVM 
    ## Number of resamples: 10 
    ## 
    ## Accuracy 
    ##          Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
    ## RF  0.5333333 0.5750000 0.5899596 0.5888087 0.6123179 0.6333333    0
    ## LVQ 0.5333333 0.5631250 0.5980353 0.5947599 0.6334879 0.6400000    0
    ## GBM 0.5099338 0.5733333 0.5986577 0.5915461 0.6076159 0.6733333    0
    ## SVM 0.5369128 0.5704857 0.5780574 0.5834705 0.6033333 0.6291391    0
    ## 
    ## Kappa 
    ##             Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
    ## RF   0.046321526 0.1354860 0.1586058 0.1587568 0.2067201 0.2536638    0
    ## LVQ  0.031543996 0.1075573 0.1732372 0.1672570 0.2461851 0.2610837    0
    ## GBM -0.003592599 0.1188102 0.1767293 0.1587341 0.1868163 0.3287671    0
    ## SVM  0.063405302 0.1120807 0.1312695 0.1464164 0.1940910 0.2351664    0

All methods perform relatively well in terms of Accuracy, but Gradient
Boosting Machine has the lowest median accuracy, same the lowest low
Kappa. SVM has also the widest range:

    # boxplots of results
    bwplot(results)

![](BTC_Test_files/figure-markdown_strict/btc_model_charts-1.png)

    # dot plots of results
    dotplot(results)

![](BTC_Test_files/figure-markdown_strict/btc_model_charts-2.png)

Best method: Learning Vector Quantization lvq
=============================================

### mean Accuracy 59,2%

### mean Kappa: 16,5%

Kappa = the amount of agreement correct by the agreement expected by
chance. In other words, how much better, the classifier is than what
would be expected by random chance.
